# Basic Example
![demo-catchball-dqn](https://raw.githubusercontent.com/algolab-inc/tensorflow-deep-q-network/master/basic/demo-catchball-dqn.gif)

## Quick Start
```
git clone https://github.com/algolab-inc/tensorflow-deep-q-network.git
cd tensorflow-deep-q-network/basic
python test.py -m models/pretrained/catchball-dqn.ckpt
```

## Usage
### Train
```
python train.py
```

### Test
```
python test.py
```

### Test with trained model
```
python test.py -m models/pretrained/catchball-dqn.ckpt
```

## Requirements
* TenforFlow
* Matplotlib
* ImageMagick (Optional)
