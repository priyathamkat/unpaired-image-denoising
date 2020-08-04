# Unpaired Image Denoising

Code for the paper titled **Unpaired Image Denoising** accepted for ICIP 2020.
## Instructions to reproduce
Module dependencies are listed in `requirements.txt`.

First download the MS COCO dataset and split it into 3 folders: `clean`, `noisy` and `test`.
### Stage 1
In this stage, the images in `clean` are used to train a flow-based model.

    cd src/glow
    python train.py --clean_path=<Path to `clean` split of COCO dataset>

### Stage 2
In this stage, a resnet is trained with inputs from the `noisy` folder with the flow-based model trained above as prior.

    cd src/
    python train.py --datsaset=<Path to full COCO dataset> --saved_flow_model=<Path to ckpt file of trained flow based model>

Testing is also done along with training (see `src/train.py`). 