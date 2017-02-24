# Melanoma_Keras
Melanoma

Folder structure:

```
Data/
    ISIC-2017_Training_Data/
              ISIC_0000000.jpg
              ...
    ISIC-2017_Training_Data_metadata.csv
    ISIC-2017_Training_Part3_GroundTruth.csv
```

Ordering:

img_preprocess.py: распределяет изображения по 3-м директориям;
conv_nn.py: запускает тренировку модели;
