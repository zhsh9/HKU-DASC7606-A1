# Notes

## code

- [x] finish task1-5, ready to train the model with training & validation dataset

## model

tain on training ds:

```bash
python train.py --coco_path ./data --output_path ./output --depth 50 --epochs 20
```

evaluate on validation ds:

```bash
python test.py --coco_path ./data --checkpoint_path ./output/model_final.pt --depth 50 --set_name 'val'
```

visualization:

```bash
python vis.py
```

generate test set predictions on testing ds:

```bash
python test.py --coco_path ./data --checkpoint_path ./output/model_final.pt --depth 50
python test_submission.py --coco_path ./data # check output format
```
