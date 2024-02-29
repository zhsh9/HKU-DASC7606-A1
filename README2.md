# Notes

## code

- [x] finish task1-5, ready to train the model with training & validation dataset

## log name

```
train_depth50_epochs20_no1           _feb29_1804.log
train,param_list      ,the Nth train ,date
```

## model1

tain on training ds:

```console
$ python train.py --coco_path ./data --output_path ./output --depth 50 --epochs 20 | tee log/train_depth50_epochs20_no1_feb29_1804.log
...
[1.5538244498526956, 1.3072959174966718, 1.1412407652892935, 1.0472031251241372, 0.9675452931248766, 0.8907449831526111, 0.8338661640735827, 0.7637128479720101, 0.7038242877057688, 0.6509080720317411, 0.6150699346582955, 0
.562676263269125, 0.5294120622373472, 0.49734498825158896, 0.462800533936073, 0.43352010441342675, 0.40314561761475687, 0.3795664301362094, 0.3545866460304737, 0.32529307411998276]
```

evaluate on validation ds:

```console
$ python test.py --coco_path ./data --checkpoint_path ./output/model_final.pt --depth 50 --set_name 'val' | tee log/valid_depth50_epochs20_no1_feb29_2034.log
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.224
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.413
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.223
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.005
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.064
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.290
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.300
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.397
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.020
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.212
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.494
```

visualization:

```bash
python vis.py
```

![visual_model1](./sources/vis_model1.png)

generate test set predictions on testing ds (Undone for this model):

```bash
python test.py --coco_path ./data --checkpoint_path ./output/model_final.pt --depth 50 --set_name 'test'
python test_submission.py --coco_path ./data # check output format
```

## model2

keep all params same but train it with more epochs: 100 -> try to find the convergence point of model training 

```console
$ python train.py --coco_path ./data --output_path ./output --depth 50 --epochs 100 | tee log/train_depth50_epochs100_no2_feb29_2107.log

$ python test.py --coco_path ./data --checkpoint_path ./output/model_final.pt --depth 50 --set_name 'val' | tee log/valid_depth50_epochs20_no1_feb29_2034.log

$ python vis.py
```

### visualization

loss curve:

## Hyper-parameter tuning

| model  | depth | optimizer                             | learning rate | drop_last, drop_out | batch size | warm-up, prior | α    | γ    |
| ------ | ----- | ------------------------------------- | ------------- | ------------------- | ---------- | -------------- | ---- | ---- |
| model2 | 50    | Adam(retinanet.parameters(), lr=1e-4) | 1e-4          | False               | 2          | 0.01           | 0.25 | 2.0  |
|        | 18    |                                       |               |                     |            |                |      |      |
|        | 34    |                                       |               |                     |            |                |      |      |
|        | 101   |                                       |               |                     |            |                |      |      |
|        |       |                                       |               |                     |            |                |      |      |

