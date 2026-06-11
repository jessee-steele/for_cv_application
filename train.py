import tensorflow as tf
print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
from tensorflow.keras import losses 

import glob
import numpy as np

from models import SFANet, SFANetUnet
from utils.preprocess import *

import os
import sys
import shutil
import inspect

import h5py as h5

import matplotlib.pyplot as plt

import json

import argparse

import sys, importlib
config_name = next((sys.argv[i + 1] for i, v in enumerate(sys.argv) if v == '--config' and i + 1 < len(sys.argv)), 'config')
config = importlib.import_module(config_name)

# Version 3: Draw gaussian on the fly
def wsahr_loss(fp_weight=config.fp_weight, reg_lambda=config.reg_weight,
               gamma=0.01, focal_weight=config.focal_weight, eps=1e-6,
               base_sigma=config.sigma):
    base_sigma = float(base_sigma)
    def _loss(y_true, concat_outputs):
        gt   = y_true[..., 0:1]          # ground-truth heatmap
        pred = concat_outputs[..., 0:1]  # predicted heatmap
        scale= concat_outputs[..., 1:2]  # predicted scale (multiplier of base_sigma)

        # --- redraw exact Gaussian using predicted sigma ---
        gt_c = tf.clip_by_value(gt, eps, 1.0)
        Db   = base_sigma * tf.sqrt(tf.maximum(-2.0 * tf.math.log(gt_c), 0.0))
        sigma_pred = base_sigma * tf.maximum(scale, eps)
        scaled_gt = tf.exp(-0.5 * tf.square(Db / sigma_pred))

        # --- asymmetric FP/FN weights + focal factor ---
        w_pos = tf.pow(gt, gamma)
        w_neg = 1.0 - w_pos
        weight = (1 - fp_weight) * w_pos * tf.abs(1.0 - pred) + fp_weight * w_neg * tf.abs(pred)

        pred_clip = tf.clip_by_value(pred, eps, 1.0 - eps)
        p_t = gt * pred_clip + (1.0 - gt) * (1.0 - pred_clip)
        focal = tf.pow(1.0 - p_t, focal_weight)

        # --- heatmap regression + scale regularization ---
        sq_err = tf.square(pred - scaled_gt)
        heatmap_loss = sq_err * (2.0 * weight) * focal
        scale_loss = tf.square(tf.math.log(sigma_pred) - tf.math.log(base_sigma))

        return tf.reduce_mean(heatmap_loss) + reg_lambda * tf.reduce_mean(scale_loss)
    return _loss


def tf_data_augment(image, targets, seed=None):
    conf, att = targets
    image = tf.cast(image, tf.float32)
    # image = tf.clip_by_value(image, 0.0, 255.0)
    conf = tf.cast(conf, tf.float32)
    att = tf.cast(att, tf.float32)

    if image.shape.rank == 2:
        image = tf.expand_dims(image, -1)
    if conf.shape.rank == 2:
        conf = tf.expand_dims(conf, -1)
    if att.shape.rank == 2:
        att = tf.expand_dims(att, -1)

    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        conf = tf.image.flip_left_right(conf)
        att = tf.image.flip_left_right(att)

    # Random 90-degree rotation
    if tf.random.uniform(()) > 0.5:
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k)
        conf = tf.image.rot90(conf, k)
        att = tf.image.rot90(att, k)

    # Random central crop and resize
    if tf.random.uniform(()) > 0.7:
        crop_frac = tf.random.uniform([], 0.7, 1.0)
        combined = tf.concat([image, conf, att], axis=-1)
        combined = tf.image.central_crop(combined, crop_frac)
        combined = tf.image.resize(combined, tf.shape(image)[0:2])

        num_image_channels = tf.shape(image)[-1]
        image = combined[..., :num_image_channels]
        conf = combined[..., num_image_channels:num_image_channels + 1]
        att = combined[..., num_image_channels + 1:]

    #  # Brightness and contrast
    # if tf.random.uniform(()) > 0.6:
    #     choice = tf.random.uniform((), minval=0, maxval=3, dtype=tf.int32)
        
    #     def _brightness():
    #         return image * tf.random.uniform((), 0.6, 1.2)
    #     def _contrast():
    #         return tf.image.random_contrast(image, lower=0.7, upper=1.3)
    #     def _gamma():
    #         g = tf.random.uniform((), 0.7, 1.3)
    #         return tf.image.adjust_gamma(image / 255.0, gamma=g) * 255.0
    #     image = tf.cond(choice == 0, _brightness, lambda: tf.cond(choice == 1, _contrast, _gamma))
    # image = tf.clip_by_value(image, 0.0, 255.0)

    return image, (conf, att)



def make_tf_dataset(images, confs, atts, batch_size=8, augment=True, shuffle=True, repeat=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, (confs, atts)))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=True)
    if repeat:
        dataset = dataset.repeat()
    if augment:
        dataset = dataset.map(tf_data_augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset



class EarlyStopping(tf.keras.callbacks.Callback):
    
    def __init__(self, file_path, monitor = 'val_loss', patience = 40, mode = 'min'):
        super().__init__()
        self.file_path = file_path
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best = float('inf') if mode == 'min' else -float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        with open(self.file_path, 'w') as f:
            f.write("Early stopping has not occurred.\n")

    def on_epoch_end(self, epoch, logs = None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        if (self.mode == 'min' and current < self.best) or (self.mode == 'max' and current > self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                with open(self.file_path, 'w') as f:
                    f.write(f"Early stopping triggered at epoch {epoch+1}.\n")

class LossHistoryCSV(tf.keras.callbacks.Callback):

    def __init__(self, csv_path, resume = False):
        super().__init__()
        self.csv_path = csv_path
        if not resume or not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline = "") as f:
                f.write("epoch,train_loss,val_loss,lr\n")
    
    def on_epoch_end(self, epoch, logs = None):
        logs = logs or {}
        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        with open(self.csv_path, 'a', newline = "") as f:
            f.write(f"{epoch+1},{train_loss},{val_loss},{lr}\n")
        print(f"Epoch {epoch+1} loss was logged to {self.csv_path}")

class LossCurvePlotter(tf.keras.callbacks.Callback):

    def __init__(self, png_path):
        super().__init__()
        self.png_path = png_path
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs = None):
        logs = logs or {}
        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        plt.figure()
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, label = "Training Loss")
        plt.plot(epochs, self.val_losses, label = "Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss Curves")
        plt.savefig(self.png_path)
        plt.close()
        print(f"Epoch {epoch+1} loss curves updated to {self.png_path}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default=config.prepared_data_path, help='path to training data hdf5 file')
    parser.add_argument('--log', default=config.log_directory, help='path to log directory')

    parser.add_argument('--lr', type=float, default=config.initial_lr, help='learning rate')
    parser.add_argument('--epochs', type=int, default=config.epoch, help='num epochs')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint if early stopping has not occurred')
    parser.set_defaults(resume=getattr(config, "resume", False))  # ✅
    parser.add_argument('--augment', action='store_true', help='Apply on-the-fly data augmentation during training')
    parser.set_defaults(augment=config.augment)
    parser.add_argument('--steps_per_epoch', type=int, default=config.steps_per_epoch,
                        help='Number of steps per epoch. If not specified, defaults to train_size // batch_size')
    parser.add_argument('--init_weights', type=str, default=None,
                    help='Path to pretrained weights (e.g., best.weights.h5)')
    parser.add_argument('--ft_lr', type=float, default=None,
                        help='Optional lower LR for fine-tuning (e.g., 1e-5)')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze common backbone layers during fine-tuning')

    parser.add_argument('--config', type=str, default='config', help='Config module name (already processed)')
    args = parser.parse_args()

    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except Exception as e:
            print(e)

    # --- Fine-tuning setup ---
    if getattr(config, "init_weights", None):
        print("==========================================")
        print("🚀 Fine-tuning model")
        print(f"   → Loaded weights from: {config.init_weights}")
        if getattr(config, "freeze_backbone", False):
            print("   → Backbone layers are FROZEN (only head/decoder will update).")
        else:
            print("   → All layers are TRAINABLE (full fine-tuning).")
        if getattr(config, "ft_lr", None):
            print(f"   → Fine-tuning LR: {config.ft_lr}")
        print("==========================================")
    else:
        print("🆕 Training from scratch (no pretrained weights).")

            
    # derive geo_dim from mode
    geo_mode = getattr(config, 'geo_mode', 'unit')  # 'unit' | 'fourier' | 'none'
    freqs    = getattr(config, 'geo_fourier_freqs', (1,2,4,8))
    if geo_mode in (None, 'none'):
        geo_dim = 0
    elif geo_mode == 'unit':
        geo_dim = 3
    elif geo_mode == 'fourier':
        geo_dim = 4 * len(freqs)   # <-- FIXED (16 for (1,2,4,8))
    else:
        raise ValueError(f"Unsupported geo_mode: {geo_mode}")
 
    # Start train
    f = h5.File(args.data, 'r')
    bands = f.attrs['bands']

    if 'val/images' in f:
        val_images = f['val/images'][:]
        val_confidence = f['val/confidence'][:]
        val_attention = f['val/attention'][:]
    else:
        print("Warning: Validation data not found in HDF5 file. Using training data as validation.")
        val_images = f['train/images'][:]
        val_confidence = f['train/confidence'][:]
        val_attention = f['train/attention'][:]
    
    preprocess = eval(f'preprocess_{bands}')
    
    model_type = config.model
    
    # Build model (same options you used in training)
    if config.model == 'vgg':
        model_builder = SFANet
    elif config.model == 'unet':
        model_builder = SFANetUnet
    else:
        raise ValueError(f"Unsupported model type: {config.model}")
        
    # Important: pass geo_mode/geo_dim so the input signature matches the weights
    model, infer_model = model_builder.build_model(
        val_images.shape[1:], preprocess_fn=preprocess,
        geo_mode=geo_mode, geo_dim=(None if geo_dim==0 else geo_dim),
        film_hidden=getattr(config,'geo_film_hidden',64),
        film_dropout=getattr(config,'geo_film_dropout',0.1),
        scale_min=getattr(config,'scale_min',0.1),
        backbone=getattr(config, 'backbone', None),
        encoder_weights=getattr(config, 'pretrain_w', None)
    )
    
    
    
    # ///////
    # Detect model inputs
    expects_two_inputs = isinstance(model.input_shape, (list, tuple)) and len(model.input_shape) == 2

    # Pick geo dataset names
    geo_ds_train = f'train/geo_{geo_mode}'
    geo_ds_val   = f'val/geo_{geo_mode}'

    # Sanity checks + load
    if expects_two_inputs and geo_mode != 'none':
        if geo_ds_train not in f or geo_ds_val not in f:
            raise RuntimeError(
                f"Model expects [image, geo] but '{geo_ds_train}' or '{geo_ds_val}' not found in HDF5. "
                "Either prepare with both geos (done) or set geo_mode='none'."
            )
        train_geos = f[geo_ds_train][:]
        val_geos   = f[geo_ds_val][:]
    else:
        train_geos = None
        val_geos   = None

    print(f"→ Using geo_mode='{geo_mode}' with geo_dim={geo_dim}; "
          f"train_geos shape: {None if train_geos is None else train_geos.shape}")
    # ///////
    
    
    # --- Load pretrained weights if provided in config ---
    if getattr(config, "init_weights", None):
        print(f"→ Loading init weights from: {config.init_weights}")
        model.load_weights(config.init_weights)

    # --- Optionally freeze backbone ---
    if getattr(config, "freeze_backbone", False):
        frozen = 0
        for layer in model.layers:
            name = layer.name.lower()
            if any(k in name for k in ['backbone', 'encoder', 'resnet', 'efficientnet', 'conv1']):
                layer.trainable = False
                frozen += 1
        print(f"🔒 Frozen {frozen} backbone layers")
        
    # --- Optionally finetune lr ---
    effective_lr = config.ft_lr if getattr(config, "ft_lr", None) else args.lr
    
    opt = Adam(effective_lr)
    model.compile(optimizer = opt, loss = [wsahr_loss(), 'binary_crossentropy'], loss_weights = [1, config.att_weight])

    print(model.summary())
    
    os.makedirs(args.log, exist_ok = True)

    # Save full config.py file to a text record
    config_filename = os.path.basename(inspect.getfile(config))
    config_save_path = os.path.join(args.log, config_filename)
    shutil.copyfile(inspect.getfile(config), config_save_path)
    
    

    print(f"✓ Config file saved to {config_save_path}")



    callbacks = []

    best_weights_path = os.path.join(args.log, 'best.weights.h5')
    callbacks.append(
        ModelCheckpoint(
            filepath = best_weights_path,
            monitor = 'val_loss',
            verbose = True,
            save_best_only = True,
            save_weights_only = True,
        )
    )
    latest_weights_path = os.path.join(args.log, 'latest.weights.h5')
    callbacks.append(
        ModelCheckpoint(
            filepath = latest_weights_path,
            monitor = 'val_loss',
            verbose = True,
            save_best_only = False,
            save_weights_only = True
        )
    )
    tensorboard_path = os.path.join(args.log, 'tensorboard')
    os.system("rm -rf " + tensorboard_path)
    callbacks.append(tf.keras.callbacks.TensorBoard(tensorboard_path))
    
    # Add learning rate scheduler
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Reduce LR
        patience=15,  # Number of epochs with no improvement
        verbose=1,
        mode='min',
        min_delta=0.0001,
        min_lr=1e-6  # Minimum learning rate
    )
    callbacks.append(reduce_lr)

    csv_file = os.path.join(args.log, "loss_history.csv")
    callbacks.append(LossHistoryCSV(csv_file, resume = args.resume))
    
    loss_curves_path = os.path.join(args.log, "loss_curves.png")
    callbacks.append(LossCurvePlotter(loss_curves_path))
    
    early_stopping_log_path = os.path.join(args.log, 'early_stopping_log.txt')
    early_stopping_callback = EarlyStopping(early_stopping_log_path, monitor = 'val_loss', patience = 45, mode = 'min')
    callbacks.append(early_stopping_callback)

    initial_epoch = 0
    if args.resume:
        if os.path.exists(latest_weights_path):
            if os.path.exists(early_stopping_log_path):
                with open(early_stopping_log_path, 'r') as f_log:
                    content = f_log.read()
                if "Early stopping triggered" in content:
                    raise Exception("Early stopping has occurred previously. Please restart training from scratch.")
            if not os.path.exists(csv_file):
                raise Exception("Loss history CSV file does not exist. Please restart training from scratch.")
            with open(csv_file, 'r') as f_csv:
                lines = f_csv.readlines()
                print(lines)
                if len(lines) <= 1:
                    raise Exception("Loss history CSV file does not contain epoch information. Please restart training from scratch.")
                last_line = lines[-1]
                try:
                    initial_epoch = int(last_line.split(',')[0])
                except Exception:
                    raise Exception("Could not parse the last epoch from CSV. Please restart training from scratch.")
            print("Resuming training from latest checkpoint.")
            model.load_weights(latest_weights_path)
            print(f"Resuming from epoch {initial_epoch}.")
        else:
            raise Exception("No latest checkpoint found. Please restart training from scratch.")


    def hdf5_generator(images, confs, atts, geos=None):
        def gen():
            for i in range(len(images)):
                image = images[i]; conf = confs[i]; att = atts[i]
                if image.ndim == 2: image = image[..., np.newaxis]
                if conf.ndim  == 2: conf  = conf[...,  np.newaxis]
                if att.ndim   == 2: att   = att[...,   np.newaxis]
                if geos is None:
                    yield image, (conf, att)
                else:
                    yield (image, geos[i]), (conf, att)
        return gen

    def make_hdf5_dataset(images, confs, atts, batch_size, augment=False, shuffle=False, repeat=False, geos=None):
        n_bands = images.shape[-1] if images[0].ndim == 3 else 1
        have_geo = geos is not None

        if have_geo:
            output_signature = (
                (tf.TensorSpec(shape=(256,256,n_bands), dtype=tf.float32),
                 tf.TensorSpec(shape=(geos.shape[1],), dtype=tf.float32)),
                (tf.TensorSpec(shape=(256,256,1), dtype=tf.float32),
                 tf.TensorSpec(shape=(256,256,1), dtype=tf.float32))
            )
        else:
            output_signature = (
                tf.TensorSpec(shape=(256,256,n_bands), dtype=tf.float32),
                (tf.TensorSpec(shape=(256,256,1), dtype=tf.float32),
                 tf.TensorSpec(shape=(256,256,1), dtype=tf.float32))
            )

        ds = tf.data.Dataset.from_generator(
            lambda: hdf5_generator(images, confs, atts, geos)(),
            output_signature=output_signature
        )

        if shuffle:
            ds = ds.shuffle(1024)
        if repeat:
            ds = ds.repeat()

        # IMPORTANT: augmentation must pass geo through unchanged when present
        if augment:
            if have_geo:
                def _map_with_geo(inputs, targets):
                    image, geo = inputs
                    image, targets = tf_data_augment(image, targets)
                    return (image, geo), targets
                ds = ds.map(_map_with_geo, num_parallel_calls=tf.data.AUTOTUNE)
            else:
                ds = ds.map(tf_data_augment, num_parallel_calls=tf.data.AUTOTUNE)

        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    
    # Decide steps_per_epoch and repeat behavior
    if args.steps_per_epoch is None:
        steps_per_epoch = None
        # int(np.ceil(len(f['train/images']) / args.batch_size))
        repeat_train = False
    else:
        steps_per_epoch = args.steps_per_epoch
        repeat_train = True  # allow looping if user manually sets steps_per_epoch

    # Create datasets
    train_dataset = make_hdf5_dataset(
        f['train/images'], f['train/confidence'], f['train/attention'],
        batch_size=args.batch_size, augment=args.augment,
        shuffle=True, repeat=repeat_train, geos=train_geos
    )
    val_dataset = make_hdf5_dataset(
        f['val/images'], f['val/confidence'], f['val/attention'],
        batch_size=args.batch_size, augment=False,
        shuffle=False, repeat=False, geos=val_geos
    )

    val_steps = int(np.ceil(len(f['val/images']) / args.batch_size))

    # Train
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        initial_epoch=initial_epoch,
        verbose=True,
        validation_steps=val_steps,
        callbacks=callbacks
    )

    

if __name__ == '__main__':
    main()