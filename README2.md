# Code

Code is [here](https://github.com/zhsh9/HKU-DASC7606-A1).

- [x] finish task1-5, ready to train the model with training & validation dataset
- [x] fulfill resnet depth option: 18,34,50,101,152
- [x] set the manual seed of torch: 3407
- [x] visualiza loss curves

# definition of logname

```
train_depth50_epochs20_no1           _feb29_1804.log
train,param_list      ,the Nth train ,date
```

# models

## model1

tain on training ds:

```console
$ python train.py --coco_path ./data --output_path ./output --depth 50 --epochs 20 | tee log/train_depth50_epochs20_no1.log
...
[1.5538244498526956, 1.3072959174966718, 1.1412407652892935, 1.0472031251241372, 0.9675452931248766, 0.8907449831526111, 0.8338661640735827, 0.7637128479720101, 0.7038242877057688, 0.6509080720317411, 0.6150699346582955, 0
.562676263269125, 0.5294120622373472, 0.49734498825158896, 0.462800533936073, 0.43352010441342675, 0.40314561761475687, 0.3795664301362094, 0.3545866460304737, 0.32529307411998276]
```

evaluate on validation ds:

```console
$ python test.py --coco_path ./data --checkpoint_path ./output/model_final.pt --depth 50 --set_name 'val' | tee log/valid_depth50_epochs20_no1.log
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.324
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

### training

keep all params same but train it with more epochs: 40 -> try to find the convergence point of model training -> which epochs to set for the furthermore models.

```console
$ python train.py --coco_path ./data --output_path ./output --depth 50 --epochs 40 | tee log/train_depth50_epochs40_no2.log
...
epoch_loss_list:
[1.5916821927888187, 1.3292744846383886, 1.1823020935879918, 1.0957840012577105, 1.01449471257451, 0.9309584555898126, 0.8588897227126313, 0.7991556330459324, 0.7447545866327961, 0.6923033849402206, 0.6518430723625375, 0.6
04410376561087, 0.56528558134006, 0.5191527205688042, 0.4878989535816542, 0.4592239510370638, 0.4343891706588993, 0.4030075690371725, 0.37277944700235105, 0.35764804361828084, 0.3348241330777097, 0.3168013905666358, 0.2976
029793196541, 0.28668593891390315, 0.27693140523977516, 0.2621651426265676, 0.24947327262296629, 0.24383938032767083, 0.23293846984958555, 0.2250983702664183, 0.2157967944728489, 0.21210157690298428, 0.19975327165599355, 0
.19877907846812307, 0.19323293236634276, 0.18541328297326648, 0.17950393673303852, 0.1767823570242929, 0.1731491775497589, 0.16610762147658217]
...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.329
```

### visualization

loss curve:

## Hyper-parameter tuning

| model      | depth | optimizer | learning rate | drop_last | batch size | warm-up, prior | α    | γ    |
| ---------- | ----- | --------- | ------------- | --------- | ---------- | -------------- | ---- | ---- |
| model1     | 50    | Adam      | 1e-4          | False     | 2          | 0.01           | 0.25 | 2.0  |
| model2     | 50    | Adam      | 1e-4          | False     | 2          | 0.01           | 0.25 | 2.0  |
| **model3** | 18    | Adam      | 1e-4          | False     | 2          | 0.01           | 0.25 | 2.0  |
| model4     | 34    | Adam      | 1e-4          | False     | 2          | 0.01           | 0.25 | 2.0  |
| model5     | 101   | Adam      | 1e-4          | False     | 2          | 0.01           | 0.25 | 2.0  |
|            |       |           |               |           |            |                |      |      |

Param position in code (if alteration is essential):

- `depth`

```console
$ python train.py --depth <depth>
$ python test.py --depth <depth>
```

- `optimizer`, `learning_rate`

```python
# train.py > main > optimizer
optimizer = optim.Adam(retinanet.parameters(), lr=1e-4)
```

- `drop_last`
  - The `drop_last` parameter in this context controls whether the data loader should discard the incomplete batch at the end of the iteration when the size of the dataset is not divisible by the batch size.
  - In practice, the choice of `drop_last` depends on your specific requirements. If you want to ensure that each batch has the same number of data points to maintain consistency in training, you might choose `drop_last=True`. However, if you want to utilize all available data, especially when the dataset is not very large, you might opt for `drop_last=False`. It is important to note, however, that retaining an incomplete batch may lead to inconsistencies in the batch normalization statistics (if batch normalization is used) for the last batch compared to the others, which could have a minor impact on training.
- `batch_size` determines the number of data samples used to compute gradients and update model weights during a training iteration. Specifically, the role of the `batch_size` parameter in this context includes:
  - **Memory Management**: The batch size directly impacts the amount of memory required during the training process. Smaller batches use less memory, allowing models to be trained on systems with limited memory. However, batches that are too small can lead to unstable training or slow convergence.
  - **Computational Efficiency**: By leveraging the parallel processing capabilities of modern hardware (like GPUs or TPUs), larger batches can improve computational efficiency. This is because the hardware can process multiple samples within a batch simultaneously, thus completing gradient calculations and weight updates more quickly.
  - **Gradient Estimation**: The samples in each batch are used to estimate the gradient at that point. Larger batches can provide a more accurate gradient estimate as they incorporate information from more data. However, batches that are too large might reduce the model's ability to generalize during training.
  - **Convergence**: Batch size can affect the convergence properties during model training. Some research suggests that an appropriate batch size can help the model converge more quickly to a better solution.
  - **Regularization Effect**: Smaller batches can sometimes have a regularization effect since each update is noisier, which can help the model avoid getting stuck in local minima.

```python
# train.py > main > sampler
sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)

# retinanet > dataloader.py > AspectRatioBasedSampler
class AspectRatioBasedSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last):
        self.batch_size = batch_size
				...

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]
```

- warm-up (`prior`)

```python
# retinanet > model.py > ResNet > model_init
def model_init(self):
  	...
    prior = 0.01
    self.classificationModel.output.weight.data.fill_(0)
    self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
    ...
```

- `α, γ`

```python
# retinanet > losses.py > FocalLoss > forward
class FocalLoss(nn.Module):
    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        ...
```

## model3,4,5

- model3
  - Based on the loss_list, we are informed that 50 epochs is not enough for network depth 18

```console
$ python train.py --coco_path ./data --output_path ./model3 --depth 18 --epochs 80 | tee log/train_depth18_epochs80_no3.log
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.362
...
epoch_loss_list:
[1.528632529783906, 1.246746529832365, 1.0978513049387086, 1.0141713309182425, 0.908193542658696, 0.8403387490923949, 0.7831078268088928, 0.731229337093633, 0.6776719326080065, 0.6424152135086341, 0.5919587297702399, 0.5572629134779371, 0.5228129406496296, 0.48738525512327596, 0.46106301398405175, 0.42910092300552083, 0.4036207692697644, 0.38645969763926163, 0.36142518085346914, 0.34390027860663536, 0.32629888006815994, 0.31176110758541487, 0.29622646571892336, 0.2785549231375912, 0.270975688630055, 0.25835850701564705, 0.2511534537573704, 0.23763869246687946, 0.22856825399469202, 0.2205726409946957, 0.222689586724558, 0.2101651065300886, 0.21022608962119008, 0.1954633732579326, 0.18856240966811952, 0.18861498967304416, 0.18163314815537315, 0.18002563769705446, 0.16797019798843937, 0.16953391215294658, 0.1666016248586463, 0.15891323118529274, 0.15243006697085898, 0.15006576055536178, 0.15150466023240328, 0.14260925523438087, 0.13658058772956233, 0.13579073711940065, 0.10504844202339195, 0.0831657922460045, 0.06969752579956777, 0.06280357125373355, 0.056226964000022174, 0.051520526260536305, 0.04750911243648111, 0.043586513231963095, 0.0411977958946817, 0.03965084551623656, 0.03728222996147724, 0.03641852081442109, 0.03492345321899661, 0.03340102444191126, 0.032040469704265065, 0.03145714819167308, 0.029539805936260887, 0.028259916604974573, 0.02796289171266377, 0.02773462125378859, 0.02691339467179835, 0.026762918414347293, 0.026478477183309583, 0.025982557757757374, 0.02613322120511631, 0.025509635560375162, 0.025345785843928464, 0.025471286296561008, 0.025652885035459067, 0.02472111577588394, 0.024803984546795575, 0.02472613626610035]

$ python test.py --coco_path ./data --checkpoint_path ./model3/model_final.pt --depth 18 --set_name 'val' | tee log/valid_depth18_epochs50_no3.log

$ python vis.py
```

- model4

```console
$ python train.py --coco_path ./data --output_path ./model4 --depth 34 --epochs 50 | tee log/train_depth34_epochs50_no4.log
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.341
epoch_loss_list:
[1.6027211780388524, 1.3493213718097041, 1.2005559017104426, 1.0817573109832335, 0.9989016679680254, 0.9243821980739673, 0.8597765068604252, 0.806570817283758, 0.7509586330004564, 0.7024351666703468, 0.6507156748603297, 0.6194083641217215, 0.583154296531804, 0.5428892699414938, 0.5085167201001226, 0.4828783282056803, 0.45397009768002616, 0.43339138117244863, 0.4030665038006864, 0.3867030331102295, 0.3644278600819292, 0.34267387191552345, 0.32291754081984614, 0.3062023756746936, 0.29318198249088673, 0.2797942441965153, 0.2705999839882271, 0.25520450626243285, 0.24268527382500762, 0.23772001985871183, 0.2299480134764261, 0.2207381194934073, 0.21444946402082526, 0.2018105147413792, 0.1983299547142723, 0.19787831321050683, 0.19110252841111772, 0.18569881874539956, 0.17913870857368538, 0.17234412436355198, 0.16710634597239826, 0.1669551119021338, 0.1568848570830768, 0.15872284962509678, 0.15080245019314356, 0.14963583615113518, 0.14378988040761478, 0.14532693679899153, 0.1016612858323593, 0.08186726589819167]

$ python test.py --coco_path ./data --checkpoint_path ./model4/model_final.pt --depth 18 --set_name 'val' | tee log/valid_depth34_epochs50_no4.log

$ python vis.py
```

- model5

```console
$ python train.py --coco_path ./data --output_path ./model5 --depth 101 --epochs 50 | tee log/train_depth101_epochs50_no5.log
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.304
epoch_loss_list:
[1.4996897089551753, 1.2040403247114242, 1.0653194290563817, 0.9758257034902028, 0.8809408868743679, 0.8240104862818802, 0.7586331694555564, 0.6976388244061019, 0.649624379909062, 0.6160755318362178, 0.5736206129897298, 0.541688928717938, 0.4977719859682082, 0.46565801993481754, 0.43757234697824154, 0.41245430582532966, 0.3843969932617986, 0.36113498351485357, 0.34414086817199085, 0.3250861821960159, 0.31000295087079127, 0.2989282929081851, 0.28059000799828393, 0.26884444194706525, 0.2534607782387945, 0.25112210249704287, 0.23778080031290416, 0.229856338058635, 0.22842503081655174, 0.21782112319396413, 0.2100386019836466, 0.2062964416200691, 0.20871497282032364, 0.19068553230329233, 0.1907811889536752, 0.1849663094892423, 0.17380056378988945, 0.17445705697585748, 0.16780385347981533, 0.15981957240132835, 0.1593968208483237, 0.1579678018172235, 0.1518478942063351, 0.15177865247388816, 0.14330497362438266, 0.14385676012909787, 0.13573541866540248, 0.13415937287380963, 0.10076992388303194, 0.08006209629041502]

$ python test.py --coco_path ./data --checkpoint_path ./model5/model_final.pt --depth 101 --set_name 'val' | tee log/valid_depth101_epochs50_no5.log

$ python vis.py
```

# Manipulate Seed & Learning Rate

**Reference**: [[2109.08203\] Torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision (arxiv.org)](https://arxiv.org/abs/2109.08203)

`torch.manual_seed(3407)` is a function used in PyTorch to set the seed for the random number generator, ensuring the repeatability of experimental results. When you set a fixed seed in your code, the random numbers generated subsequently will be predictable. This means that each time you run the same code, operations involving random number generation will yield the same results. This is very useful for debugging and for comparing the performance of different models, as it removes the variability introduced by randomness.

To ensure the reproducibility of model training, you can set a fixed random seed in your training script. This will help ensure that you get the same results each time you run the script because the initialization of weights, dataset splitting, data augmentation, and so forth will all be based on this random seed, thus maintaining consistency. You can set the random seed by adding some code at the beginning of the `main` function. This includes setting the seed for `numpy`, `torch`, `random`, and any other dependent libraries that may be used.

Note that even with a random seed set, non-deterministic operations on the GPU may still introduce some randomness. If you need to ensure as much reproducibility as possible, you might consider setting `torch.backends.cudnn.deterministic` to `True`, but this could sacrifice some performance. Additionally, multithreading (such as multiple worker threads in data loaders) may also cause some variability, and setting a random seed may not completely eliminate it.

| model  | seed   | depth | optimizer | learning rate | prior | α    | γ    |
| ------ | ------ | ----- | --------- | ------------- | ----- | ---- | ---- |
| model3 | random | 18    | Adam      | 1e-4          | 0.01  | 0.25 | 2.0  |
| model6 | 3407   | 18    | Adam      | 1e-4          | 0.01  | 0.25 | 2.0  |
| model7 | 3407   | 18    | Adam      | 1e-3          | 0.01  | 0.25 | 2.0  |
| model8 | 3407   | 18    | Adam      | 1e-5          | 0.01  | 0.25 | 2.0  |
| model9 | 3407   | 101   | Adam      | 1e-4          | 0.01  | 0.25 | 2.0  |

Setting the `learning rate` is a crucial decision in machine learning and deep learning as it significantly affects the efficiency and outcome of model training. The learning rate determines the magnitude of model weight updates during each iteration. If the learning rate is too high, it may cause the model to overshoot the optimum point while minimizing the loss function, thus preventing convergence; if the learning rate is too low, model training can be very slow and may get stuck at local minima.

1. Setting the initial learning rate:

   - **Empirical value**: Start with commonly recommended values (for example, 3e-4 is a typical initial learning rate for the Adam optimizer, while SGD often requires higher values, such as 1e-2 or 1e-1).
   - **Learning rate scheduling**: Set a higher initial learning rate and gradually decrease it during training (for instance, using strategies like learning rate decay, step decay, or cosine annealing).

2. Testing and adjusting:

   - **Learning rate range test**: Begin with a small learning rate, gradually increase it, and record the loss values to find the range where the loss decreases most rapidly.
   - **Cross-validation**: Train multiple times using different learning rate values and verify which one performs the best.

## model6

```console
$ python train.py --coco_path ./data --output_path ./model6 --depth 18 --seed 3407 --learning_rate 0.0001 --epochs 20 | tee log/train_depth18_seed3407_lr1e-4_epochs20.log
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.367
...
epoch_loss_list:
[1.4186146195948592, 1.0810809980227252, 0.9426405721703383, 0.8578932251869225, 0.7786534885017891, 0.7206757530482031, 0.6638822823205567, 0.6155609383915118, 0.5721351678270524, 0.5387147602544525, 0.5031234409777433, 0.4656750471001183, 0.43533992793000353, 0.4116135801254647, 0.38483528913856724, 0.3696655480697106, 0.3465075497625498, 0.32951304744049087, 0.31531340801560387, 0.2939375856942607]
```

## model7

```console
$ python train.py --coco_path ./data --output_path ./model7 --depth 18 --seed 3407 --learning_rate 0.001 --epochs 20 | tee log/train_depth18_seed3407_lr1e-3_epochs20.log
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.010
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.033
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.003
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.020
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.053
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.170
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.220
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.002
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.302
epoch_loss_list:
[87.95857101913512, 14.982650478700483, 1.560555976852188, 1.536893064288173, 22.14822560812779, 56.4836165021489, 2.642567965576029, 2.061367801265923, 2.0526319881827813, 6.897480560334649, 192460.9973601208, 1.953764165
186976, 1.940602258141116, 1.8743701384175482, 1.88869419881678, 455.1726897481506, 1.8737407679867557, 1.8823565633043529, 1.8701253193687266, 15.167173102908716]
```

## model8

```console
$ python train.py --coco_path ./data --output_path ./model8 --depth 18 --seed 3407 --learning_rate 0.00001 --epochs 50 > log/train_depth18_seed3407_lr1e-5_epochs50_no8.log
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.361
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.669
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.345
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.070
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.200
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.432
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.355
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.467
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.479
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.134
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.341
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.549
epoch_loss_list:
[1.3485985781381449, 1.0428863803115416, 0.9080038053372245, 0.8198696776755213, 0.7462012804457991, 0.6849284630972804, 0.6314117278258397, 0.5838938797640754, 0.5408082131853723, 0.5003792778772163, 0.47037841241044087, 0.4360208642837687, 0.4126990730324247, 0.3900142498974373, 0.3647483795749273, 0.34710889965603786, 0.3285140351531661, 0.3097159046291073, 0.300746954141522, 0.28417240203116234, 0.2706226208320577, 0.25768760743543623, 0.24772578569891768, 0.24061031865073354, 0.23440500140908782, 0.22438549321680146, 0.21360355310369372, 0.20982268031715878, 0.20265136179184234, 0.19518770426731763, 0.1902201386538928, 0.18584581637043712, 0.1828563255001241, 0.17607332579974877, 0.17221467088225964, 0.1696188291883111, 0.16682294164564254, 0.16123330928421925, 0.15749495586941356, 0.1559871606928421, 0.15146176567792483, 0.15084946997955032, 0.14413518245390608, 0.14543778893141823, 0.14053164449095082, 0.1371312429926791, 0.13400624088547008, 0.13404743663057334, 0.10965902876387548, 0.09319095793326833]
```

## final model (model6)

depth18, epochs72, learning rate 1e-4, seed 3407:

```console
$ python train.py --coco_path ./data --output_path ./model --depth 18 --seed 3407 --learning_rate 0.00001 --epochs 72
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.405
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.612
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.433
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.033
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.258
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.484
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.489
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.490
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.055
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.337
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.571
[1.4318664156076477, 1.1001639135005905, 0.951171794979591, 0.8669462496136117, 0.7826694082322083, 0.7231154088664242, 0.6712237381207662, 0.6253459720513014, 0.5767641523046287, 0.5353238649490312, 0.5019147138837285, 0.46833128531969437, 0.43027442954773976, 0.4057700016836482, 0.3837973687093793, 0.3491113050070804, 0.3365979572010087, 0.31390098533703115, 0.30307657208557676, 0.28886881692144345, 0.27685422390171394, 0.2594804616688978, 0.2531161196529865, 0.24239540178563415, 0.23492269075702965, 0.22669022654249207, 0.2124996097157086, 0.20594180111340651, 0.19773296042396796, 0.19331663411374636, 0.1847791956429641, 0.177922433542865, 0.1754182980315188, 0.176792214397253, 0.16507695097564243, 0.16730668246892727, 0.16016591913030137, 0.15313682938361262, 0.15015995093276652, 0.1490433017803928, 0.13994556185092746, 0.1417805902278564, 0.1370867475001948, 0.13476632985719075, 0.12725584065233628, 0.12609577884413595, 0.12433190355590713, 0.12400001437791924, 0.09044798064654268, 0.07121378269897202, 0.0622335343409007, 0.05580204240446104, 0.049807331673287616, 0.04686529077607845, 0.042925731446704, 0.04024196644111969, 0.03772618067744151, 0.03557270235844015, 0.03374018094146727, 0.03265945929218098, 0.03141849596735354, 0.030416728041820433, 0.02952122467929336, 0.028430486348637297, 0.027613410924187325, 0.02634765021457139, 0.02603730824537663, 0.025673939840039458, 0.025331346891035774, 0.02523221916200109, 0.024656759558280374, 0.02444344596624719]
```

```console
$ python test.py --coco_path ./data --checkpoint_path ./output/model_final.pt --depth 18 --set_name 'val' | tee log/valid_model_final.log
Accumulating evaluation results...
DONE (t=0.38s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.405
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.612
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.433
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.033
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.258
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.484
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.489
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.490
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.055
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.337
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.571
```

```console
$ python test.py --coco_path ./data --checkpoint_path ./output/model_final.pt --depth 18 --set_name 'test' | tee log/test_model_final.log
loading annotations into memory...
Done (t=0.02s)
creating index...
index created!
Num test images: 643
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 643/643 [03:16<00:00,  3.28it/s]
Loading and preparing results...
DONE (t=0.03s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.53s).
Accumulating evaluation results...
DONE (t=0.19s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.073
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.160
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.057
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.006
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.112
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.172
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.192
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.192
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.022
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.268
```

![model6](./retinanet.png)

# Report

## Codes

### calc_iou

The selected code is a Python function that calculates the Intersection over Union (IoU) of two bounding boxes. The IoU is a measure of the overlap between two bounding boxes. It's often used in object detection tasks to measure the accuracy of the predicted bounding boxes against the ground truth.

The function [`calc_iou`](command:_github.copilot.openSymbolInFile?%5B%22retinanet%2Flosses.py%22%2C%22calc_iou%22%5D "retinanet/losses.py") takes two arguments, [`a`](command:_github.copilot.openSymbolInFile?%5B%22retinanet%2Flosses.py%22%2C%22a%22%5D "retinanet/losses.py") and [`b`](command:_github.copilot.openSymbolInFile?%5B%22retinanet%2Flosses.py%22%2C%22b%22%5D "retinanet/losses.py"), which are PyTorch tensors representing the coordinates of two bounding boxes in the format [x1, y1, x2, y2].

The function first calculates the width ([`iw`](command:_github.copilot.openSymbolInFile?%5B%22retinanet%2Flosses.py%22%2C%22iw%22%5D "retinanet/losses.py")) and height ([`ih`](command:_github.copilot.openSymbolInFile?%5B%22retinanet%2Flosses.py%22%2C%22ih%22%5D "retinanet/losses.py")) of the intersection of the two bounding boxes. The `torch.unsqueeze` function is used to add an extra dimension to the tensor `a[:, 2]` and `a[:, 3]` (the x_max and y_max of bounding box [`a`](command:_github.copilot.openSymbolInFile?%5B%22retinanet%2Flosses.py%22%2C%22a%22%5D "retinanet/losses.py")) and `a[:, 0]` and `a[:, 1]` (the x_min and y_min of bounding box [`a`](command:_github.copilot.openSymbolInFile?%5B%22retinanet%2Flosses.py%22%2C%22a%22%5D "retinanet/losses.py")). The `torch.min` and `torch.max` functions are used to find the coordinates of the intersection of the two bounding boxes.

The `torch.clamp` function is then used to ensure that the intersection width and height are not less than zero. This is necessary because if the two bounding boxes do not overlap, the calculated intersection width and height could be negative.

The intersection area is then calculated as the product of the intersection width and height.

The area of bounding box [`b`](command:_github.copilot.openSymbolInFile?%5B%22retinanet%2Flosses.py%22%2C%22b%22%5D "retinanet/losses.py") is calculated as the product of its width and height. The union area ([`ua`](command:_github.copilot.openSymbolInFile?%5B%22retinanet%2Flosses.py%22%2C%22ua%22%5D "retinanet/losses.py")) of the two bounding boxes is calculated as the sum of the area of bounding box [`a`](command:_github.copilot.openSymbolInFile?%5B%22retinanet%2Flosses.py%22%2C%22a%22%5D "retinanet/losses.py") and the area of bounding box [`b`](command:_github.copilot.openSymbolInFile?%5B%22retinanet%2Flosses.py%22%2C%22b%22%5D "retinanet/losses.py"), minus the intersection area. The `torch.unsqueeze` function is used again to add an extra dimension to the area of bounding box [`a`](command:_github.copilot.openSymbolInFile?%5B%22retinanet%2Flosses.py%22%2C%22a%22%5D "retinanet/losses.py") before adding it to the area of bounding box [`b`](command:_github.copilot.openSymbolInFile?%5B%22retinanet%2Flosses.py%22%2C%22b%22%5D "retinanet/losses.py").

The `torch.clamp` function is used again to ensure that the union area is not less than a very small positive number (1e-8). This is to prevent division by zero in the next step.

Finally, the IoU is calculated as the ratio of the intersection area to the union area, and this value is returned by the function.

In summary, this function calculates the Intersection over Union (IoU) of two bounding boxes, which is a common metric in object detection tasks.

# Final Model Table

| model  | seed   | epochs | depth | optimizer | learning rate | prior | α    | γ    | mAP   |
| ------ | ------ | ------ | ----- | --------- | ------------- | ----- | ---- | ---- | ----- |
| model1 | random | 20     | 50    | Adam      | 1e-4          | 0.01  | 0.25 | 2.0  | 0.324 |
| model2 | random | 50     | 50    | Adam      | 1e-4          | 0.01  | 0.25 | 2.0  | 0.329 |
| model3 | random | 50     | 18    | Adam      | 1e-4          | 0.01  | 0.25 | 2.0  | 0.362 |
| model4 | random | 50     | 34    | Adam      | 1e-4          | 0.01  | 0.25 | 2.0  | 0.341 |
| model5 | random | 50     | 101   | Adam      | 1e-4          | 0.01  | 0.25 | 2.0  | 0.304 |
| model6 | 3407   | 20     | 18    | Adam      | 1e-4          | 0.01  | 0.25 | 2.0  | 0.367 |
| model7 | 3407   | 20     | 18    | Adam      | 1e-3          | 0.01  | 0.25 | 2.0  | 0.010 |
| model8 | 3407   | 50     | 18    | Adam      | 1e-5          | 0.01  | 0.25 | 2.0  | 0.361 |
| final  | 3407   | 72     | 18    | Adam      | 1e-5          | 0.01  | 0.25 | 2.0  | 0.405 |
