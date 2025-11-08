# Refactor to Replace RealESRGAN with SwinIR

## Tasks
- [x] Create directory `models/swinir/` and file `models/swinir/inference_swinir.py` with SwinIRUpscaler class
- [x] Update `app/config.py` to add SwinIR paths and remove RealESRGAN paths
- [x] Update `app/model_wrappers.py` to replace load_realesrgan with load_swinir and rename apply_realesrgan to apply_swinir
- [x] Update `app/restore_pipeline.py` to call apply_swinir instead of apply_realesrgan
- [x] Update `requirements.txt` to remove realesrgan and add basicsr
- [x] Handle `models/realesrgan/inference_realesrgan.py` (leave as is since not used)

## Followup Steps
- [ ] Download SwinIR x4 model to `models/swinir/weights/SwinIR_x4.pth`
- [ ] Install updated dependencies: `pip install -r requirements.txt`
- [ ] Test the API with an image to ensure upscaling works with SwinIR
- [ ] Verify output image is improved and API returns correct response
