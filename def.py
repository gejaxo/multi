    def process(self, p:Processing, 
            enabled:bool, 
            encoder_tile_size:int, decoder_tile_size:int, 
            vae_to_gpu:bool, fast_decoder:bool, fast_encoder:bool, color_fix:bool
        ):
        
        # for shorthand
        vae = p.sd_model.first_stage_model
        encoder = vae.encoder
        decoder = vae.decoder

        # undo hijack if disabled (in cases last time crashed)
        if not enabled:
            if self.hooked:
                if isinstance(encoder.forward, VAEHook):
                    encoder.forward.net = None
                    encoder.forward = encoder.original_forward
                if isinstance(decoder.forward, VAEHook):
                    decoder.forward.net = None
                    decoder.forward = decoder.original_forward
                self.hooked = False
            return

        if devices.get_optimal_device_name().startswith('cuda') and vae.device == devices.cpu and not vae_to_gpu:
            print("[Tiled VAE] warn: VAE is not on GPU, check 'Move VAE to GPU' if possible.")

        # do hijack
        kwargs = {
            'fast_decoder': fast_decoder, 
            'fast_encoder': fast_encoder, 
            'color_fix':    color_fix, 
            'to_gpu':       vae_to_gpu,
        }

        # save original forward (only once)
        if not hasattr(encoder, 'original_forward'): setattr(encoder, 'original_forward', encoder.forward)
        if not hasattr(decoder, 'original_forward'): setattr(decoder, 'original_forward', decoder.forward)

        self.hooked = True
        
        encoder.forward = VAEHook(encoder, encoder_tile_size, is_decoder=False, **kwargs)
        decoder.forward = VAEHook(decoder, decoder_tile_size, is_decoder=True,  **kwargs)

    def postprocess(self, p:Processing, processed, enabled:bool, *args):
        if not enabled: return

        vae = p.sd_model.first_stage_model
        encoder = vae.encoder
        decoder = vae.decoder
        if isinstance(encoder.forward, VAEHook):
            encoder.forward.net = None
            encoder.forward = encoder.original_forward
        if isinstance(decoder.forward, VAEHook):
            decoder.forward.net = None
            decoder.forward = decoder.original_forward
