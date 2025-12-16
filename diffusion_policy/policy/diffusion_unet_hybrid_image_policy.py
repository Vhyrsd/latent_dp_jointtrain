import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
import diffusion_policy.model.vision.crop_randomizer as dmvc

# =============================================================================
# 1. 新增：动作轨迹 VAE (Variational Autoencoder)
# =============================================================================
class TrajectoryVAE(nn.Module):
    def __init__(self, action_dim, latent_dim, hidden_dim=256):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # Encoder: (B, T, Action_Dim) -> (B, T, Latent_Dim * 2) [mu, logvar]
        # 使用 1D 卷积保持时序结构，同时压缩特征维度
        self.encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, latent_dim * 2) # 输出 mu 和 logvar
        )

        # Decoder: (B, T, Latent_Dim) -> (B, T, Action_Dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim)
        )

    def encode(self, x):
        # x: [B, T, Action_Dim]
        # output: dist (Normal), mu, logvar
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        # 限制 logvar 范围防止梯度爆炸
        logvar = torch.clamp(logvar, -10, 10)
        std = torch.exp(0.5 * logvar)
        dist = torch.distributions.Normal(mu, std)
        return dist, mu, logvar

    def decode(self, z):
        # z: [B, T, Latent_Dim]
        return self.decoder(z)

    def forward(self, x):
        dist, mu, logvar = self.encode(x)
        z = dist.rsample() # 重参数化采样
        recon_x = self.decode(z)
        return recon_x, mu, logvar


# =============================================================================
# 2. 修改后的 Latent Diffusion Policy
# =============================================================================
class DiffusionUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict, 
            noise_scheduler: DDPMScheduler, 
            horizon, 
            n_action_steps, 
            n_obs_steps, 
            num_inference_steps=None, 
            obs_as_global_cond=True, 
            crop_shape=(76, 76), 
            diffusion_step_embed_dim=256, 
            down_dims=(256,512,1024), 
            kernel_size=5, 
            n_groups=8, 
            cond_predict_scale=True, 
            obs_encoder_group_norm=False, 
            eval_fixed_crop=False, 
            
            # --- 新增 LDM 参数 ---
            latent_dim=16,          # 潜空间维度
            vae_kl_weight=0.0001,   # KL 散度权重
            # --------------------
            
            **kwargs):
        super().__init__()

        # -------------------------------------------------------------------------
        # 1. 解析参数
        # -------------------------------------------------------------------------
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.vae_kl_weight = vae_kl_weight
        
        obs_shape_meta = shape_meta['obs']
        obs_config = { 'low_dim': [], 'rgb': [], 'depth': [], 'scan': [] }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # -------------------------------------------------------------------------
        # 2. 初始化 Robomimic 视觉编码器 (Obs Encoder)
        # -------------------------------------------------------------------------
        config = get_robomimic_config(
            algo_name='bc_rnn', hdf5_type='image', task_name='square', dataset_type='ph')
        
        with config.unlocked():
            config.observation.modalities.obs = obs_config
            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        ObsUtils.initialize_obs_utils_with_config(config)
        policy = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device='cpu',
        )
        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features//16, num_channels=x.num_features)
            )
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )
        obs_feature_dim = obs_encoder.output_shape()[0]

        # -------------------------------------------------------------------------
        # 3. 初始化 VAE (Action Encoder/Decoder) - [新增]
        # -------------------------------------------------------------------------
        self.vae = TrajectoryVAE(
            action_dim=action_dim,
            latent_dim=latent_dim
        )

        # -------------------------------------------------------------------------
        # 4. 初始化 Diffusion Model (U-Net)
        # -------------------------------------------------------------------------
        # 注意: input_dim 变成了 latent_dim
        input_dim = latent_dim 
        global_cond_dim = None
        if obs_as_global_cond:
            # 输入只有 latent，观测作为 global condition
            global_cond_dim = obs_feature_dim * n_obs_steps
        else:
            # 这种混合模式在 LDM 比较少见，通常建议 LDM 使用 global cond
            # 如果非要用，input 维度是 latent_dim + obs_dim (如果obs也压缩) 或者保持 obs 不变
            # 为了简化，这里 LDM 模式强制推荐 obs_as_global_cond=True
            raise NotImplementedError("Latent Diffusion currently recommends obs_as_global_cond=True")

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.num_inference_steps = num_inference_steps
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))

    # =============================================================================
    # 核心修改：conditional_sample (推理)
    # =============================================================================
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        # 1. 初始化 Latent 噪声
        # shape: (B, T, Latent_Dim)
        B = condition_data.shape[0]
        T = self.horizon
        latent_shape = (B, T, self.latent_dim)
        
        # 从高斯分布采样初始潜变量
        latents = torch.randn(
            size=latent_shape, 
            dtype=condition_data.dtype, 
            device=condition_data.device,
            generator=generator
        )

        # 2. 设置时间步
        scheduler.set_timesteps(self.num_inference_steps)

        # 3. 扩散去噪循环 (在 Latent Space 进行)
        for t in scheduler.timesteps:
            # LDM 中通常不在 latent 空间做强制 mask (inpainting)，
            # 除非我们也训练了一个 latent mask generator。
            # 这里我们仅依赖 global_cond (观测) 来生成动作。
            
            # 预测噪声 (输入是 Latent)
            model_output = model(latents, t, local_cond=local_cond, global_cond=global_cond)

            # 采样一步: z_t -> z_{t-1}
            latents = scheduler.step(
                model_output, t, latents, 
                generator=generator,
                **kwargs
            ).prev_sample

        # 4. 解码: Latent -> Action
        # latents: [B, T, Latent_Dim] -> actions: [B, T, Action_Dim]
        action_pred = self.vae.decode(latents)

        return action_pred

    # =============================================================================
    # 核心修改：predict_action (对外接口)
    # =============================================================================
    def predict_action(self, obs_dict: dict) -> dict:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        
        # 1. Normalize observation
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        
        # 2. Encode Observation
        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
        else:
            raise NotImplementedError("LDM implementation requires obs_as_global_cond=True")

        # 3. 运行 Latent Sampling
        # 注意：这里我们不需要像原版那样构造一个混合的 condition_data (action+obs)，
        # 因为 LDM 分开了 obs (condition) 和 action (generation target)。
        # 传入空的 condition_data 只是为了获取 shape device 信息，或者保持接口兼容。
        dummy_cond_data = torch.zeros((B, T, self.action_dim), device=self.device, dtype=self.dtype)
        dummy_cond_mask = torch.zeros_like(dummy_cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(
            dummy_cond_data, 
            dummy_cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs
        )
        
        # 4. Unnormalize action
        action_pred = self.normalizer['action'].unnormalize(nsample)
        
        # 5. Extract action steps
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # =============================================================================
    # 核心修改：compute_loss (训练)
    # =============================================================================
    def compute_loss(self, batch):
        # 1. Normalize Action
        # 只归一化 action，因为 obs 通常包含图像，图像不需要 LinearNormalizer
        n_actions = self.normalizer['action'].normalize(batch['action'])
        
        # 2. Handle Obs
        # 对于 Obs，我们通常只对低维状态（agent_pos 等）做归一化
        # 如果你的 obs 包含图像，直接传给 normalizer 可能会报错，或者 normalizer 里没有图像的统计数据
        # 这里我们假设 obs 已经被 Dataset 处理好了，或者我们手动处理
        
        # 尝试使用 normalize，如果不行，说明 obs 不需要归一化或者 key 不对
        try:
            n_obs = self.normalizer.normalize(batch['obs'])
        except KeyError:
             # 如果 normalizer['obs'] 不存在，可能 obs 不需要归一化（例如全是图像）
             # 或者 key 并不是 'obs' 而是分散的
             n_obs = batch['obs']

        actions = n_actions
        obs = n_obs

        # 2. Encode Observation (Condition)
        B = actions.shape[0]
        T = actions.shape[1]
        
        if self.obs_as_global_cond:
            this_nobs = dict_apply(obs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(B, -1) # (B, obs_feat * n_obs_steps)
            local_cond = None
        else:
             raise NotImplementedError("LDM requires obs_as_global_cond=True")

        # 3. VAE Encoding: Action -> Latent
        # dist, mu, logvar = self.vae.encode(actions)
        # latents = dist.rsample() 
        # 我们同时需要计算 VAE 的 loss，所以调用 forward 得到重构结果
        recon_actions, mu, logvar = self.vae(actions)
        
        # 使用重参数化后的 latent 进行扩散训练
        # 为了保持梯度，重新采样一次或者直接利用 forward 中间结果
        # 这里为了清晰，手动重采样一次给 diffusion 使用 (或者修改VAE forward返回z)
        std = torch.exp(0.5 * logvar)
        dist = torch.distributions.Normal(mu, std)
        latents = dist.rsample() # [B, T, Latent_Dim]

        # 4. Sample noise for Diffusion
        noise = torch.randn(latents.shape, device=latents.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (B,), device=latents.device
        ).long()

        # 5. Add noise to Latents (Forward Diffusion)
        noisy_latents = self.noise_scheduler.add_noise(
            latents, noise, timesteps)

        # 6. Predict noise (Diffusion Model)
        pred = self.model(noisy_latents, timesteps, local_cond=local_cond, global_cond=global_cond)

        # 7. 计算 Loss
        
        # A. Diffusion Loss (MSE between predicted noise and real noise)
        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = latents
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
            
        diff_loss = F.mse_loss(pred, target, reduction='none')
        diff_loss = diff_loss.mean()

        # B. VAE Reconstruction Loss (MSE between actions and recon_actions)
        vae_recon_loss = F.mse_loss(recon_actions, actions, reduction='mean')

        # C. VAE KL Divergence Loss
        # KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss = kld_loss / (B * T * self.action_dim) # Normalize

        # Total Loss
        loss = diff_loss + 0.5 * vae_recon_loss + self.vae_kl_weight * kld_loss
        
        # 可选：打印各项 loss 以供调试 (在 env_runner 中可能看不到，但 compute_loss 返回 dict 更好)
        # 这里为了兼容原始接口只返回标量 loss，若要监控请在外部记录
        return loss, diff_loss, vae_recon_loss, kld_loss

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
