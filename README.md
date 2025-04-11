# SENvT

```bash
conda create -n <env name> python==3.12
conda activate <env name>
pip install -r req.txt
```

```bash
python pretrain.py
```

```bash
python downstream.py \
    --ckpt=<pre-trained model path> \
    --dataset=pamap \
```


