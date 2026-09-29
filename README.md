# ANPR-Cam
Real-time vehicle and person detection with licence-plate reading on Android, built with CameraX + TensorFlow Lite + Jetpack Compose.

## What it does.

### Live preview.
Live preview from the back camera, full screen. Vehicles are boxed in green, people in
blue, plates in yellow, and the plate string is drawn above the vehicle's own box as well
as above the plate itself.
### Detection.
TensorFlow Lite EfficientDet-Lite (COCO, int8) on every analysis frame.
Two models ship with the app:
  - Fast — EfficientDet-Lite0, 320×320, ~42 ms/frame on a desktop CPU.
  - Accurate — EfficientDet-Lite2, 448×448, ~80 ms/frame, noticeably better on distant cars.

### Letterbox preprocessing.
The frame is scaled with its aspect ratio preserved and padded
with grey 114 to the model's square input, then the predicted boxes are mapped back through
the same transform. This is not cosmetic: resizing a 720×1280 portrait frame straight to a
square squashes it 1.78× and the detector stops recognising cars altogether.

### One-pass rotate + letterbox.
  The sensor rotation is applied to the canvas, not to a
  full-resolution copy of the frame, and the result is packed into the input tensor with a single
  bulk write. See "The frame budget" below for why that matters.

### Temporal smoothing.
  BoxSmoother turns the stream of independent per-frame guesses into
  tracks: IoU-matched, EMA-eased, shown only after two confirmations, and held for ~0.5 s of
  misses so a real car never blinks out while a one-frame false positive never appears at all.

### Plate localisation:
  Classical CV inside each vehicle box (vertical Sobel edges → Otsu →
  morphological closing → connected components → geometry filter). No model file needed.
  Candidates are ranked by box area, not detector confidence — a plate is only legible when
  the car is close, so the largest box is the right one to spend OCR on.

### Plate reading:
  ML Kit's bundled Latin text recogniser on the plate crop, then a
  Ukrainian-format grammar matcher.
  
 
 
## And much more (but im lazy writing README)

  LICENSE / УСЛОВИЯ ИСПОЛЬЗОВАНИЯ

  Copyright (c) 2026 norocommod. Все права защищены.
  НАСТОЯЩИЙ МАТЕРИАЛ ПРЕДОСТАВЛЯЕТСЯ СТРОГО ДЛЯ ЛИЧНОГО ПОЛЬЗОВАНИЯ.

  Разрешено:
  - Использовать материал (код, модель, файлы) исключительно в своих личных, некоммерческих целях.
  Категорически запрещено:
  1.Перераспространять, перезаливать, передавать или публиковать данные файлы на любых сторонних ресурсах, сайтах или мессенджерах.
  2.Продавать, сдавать в аренду или использовать материал в любых коммерческих целях.
  3.Выдавать материал за свой, удалять или изменять указание авторства и копирайты.
  4.Вносить изменения, модифицировать или создавать производные работы на основе данного материала для последующей публикации.



  TERMS OF USE / LICENSE AGREEMENT

  Copyright (c) 2026 norocommod. All rights reserved.

  THIS MATERIAL IS PROVIDED STRICTLY FOR PERSONAL USE ONLY.

  Permitted:
   - Using the material (code, models, files) solely for your own personal, non-commercial purposes.
  Strictly Prohibited:
  1. Redistributing, reuploading, transferring, or publishing these files on any third-party resources, websites, or messaging platforms.
  2. Selling, renting, licensing, or using the material for any commercial purposes.
  3. Claiming ownership of the material, or removing/modifying any copyright notices or attribution.
  4. Modifying, adapting, or creating derivative works based on this material for subsequent public distribution.
