import os
from utils.generic_utils import n2t


# Configuration for tree detection
seed = 123
bands = 'RGBN' # options: 'RGB', 'RGBN'
reproject_resolution = 1 # in meter'
resample_method = 'mixed' # nearest, bilinear or mixed
res_name = n2t(reproject_resolution)
suffix = '_FINETUNE'
version = ''

### Data organization ###x
main_directory = '/home/rxv7nr/us-tree-detection-geo-share/' # CHANGED
# input_directory = os.path.join(main_directory, 'data_2018to2023_ALL')
# image_folder_name = f'images{suffix}'
# label_file_name = 'labels_pt_v5.gpkg'
input_directory = os.path.join(main_directory, 'dl_project_input_data')
image_folder_name = f'images'
label_file_name = '4.28.26_dead_tree_labels.gpkg'
#label_polygon_file_name = 'labels_polygon.gpkg'
#label_polygon_eval_file_name = 'labels_polygon_eval_v4.gpkg'
output_directory = os.path.join(main_directory, f'organized_{res_name}m_{resample_method}{suffix}')
tile = True # Tile to 256 x 256 patch size

### Tran/val/test split ###
train_val_ratio = 0.9 #9:1 ratio for now since small sample size (to run only test sample, set this to 0.5 and change test size to 2 less than your total num of images).
test_size = 8 #test on 8 randomly selected images (using a seed, so same selected each time) (CHANGED TO 74 to do *almost* ALL images)
split = 5
split_method = 'wasserstein' # 'wasserstein' or 'random'
test_fname = "" #os.path.join(input_directory, f'test_{res_name}m_2010to2025.txt')
# test_fname = os.path.join(input_directory, f'') force to do random selection

### Data preparation ###
sigma = 3
prepared_split = 'split1' #can range from split1 to split5. Different subsets of training/validation data to avoid lucky fitting.
data_frac = 0.1  # fraction of dataset to use, from 0 to 1

# --- Geo embedding options ---
geo_mode = "unit"        # options: "none", "unit", "fourier"
geo_fourier_freqs = (1, 2, 4, 8)   # used if geo_mode == "fourier"
geo_film_hidden = 64
geo_film_dropout = 0.1
geo_l2 = 1e-4            # small L2 on the geo MLP
use_cross_attention = False  # keep False for now (fiLM only)

prepared_output_directory = main_directory
prepared_fname = f'prepared_{res_name}m_{resample_method}_{sigma}sd_{prepared_split}_frac{n2t(data_frac)}{suffix}'
prepared_data_path = os.path.join(prepared_output_directory, f'{prepared_fname}.hdf5')

### Model directories ###
saved_model_directory = '/home/rxv7nr/us-tree-detection-geo-share/saved_models'

### Model parameters ###
model = "vgg"  # options: 'vgg', 'resnet', 'efficientnet', 'unet'
backbone = None
pretrain_w = None # 'imagenet' or None
steps_per_epoch = None
epoch = 1_000_000
batch_size = 12
initial_lr = 1e-4
fp_weight = 0.3 # False-positive focal weight (Direct more towards reducing false-positives)
att_weight = 0.1 # Attention-head weight
focal_weight = 1.5 # Focal weight
reg_weight = 0 # Regularizor weight
scale_min = 0.5 # Reguarizor scale map
augment = True # Apply on-the-fly data augmentation during training
resume = False # Resume training from latest checkpoint if early stopping has not occurred

# Fine-tuning options
init_weights_log_folder = "log_vgg_bs12_speNone_fp0d3_att0d1_reg0_foc1d5_scl0d5_aug_unit_prepared_1m_mixed_3sd_split1_frac1_BASE_scratch_v3" # pre-trained model weights
init_weights = None if init_weights_log_folder is None else os.path.join(saved_model_directory, init_weights_log_folder, "best.weights.h5")
ft_lr        = 1e-4        # e.g., 1e-5
freeze_backbone = False    # True to freeze encoder/backbone layers

# Build model name
fp_str = f"fp{n2t(fp_weight)}"
att_str = f"att{n2t(att_weight)}"
foc_str = f"foc{n2t(focal_weight)}"
reg_str = f"reg{n2t(reg_weight)}"
scl_str = f"scl{n2t(scale_min)}"
aug_str = "aug" if augment else "noaug"
model_str = model[:3].lower()
if model == "unet":
    backbone_str = backbone[:3].lower() if backbone else ""
    model_str = f"{model.lower()}{backbone_str}"
    if pretrain_w == "imagenet":
        model_str += "pt"
if init_weights is None:
    ft_tag = "scratch"
elif freeze_backbone:
    ft_tag = f"freeze"
else:
    ft_tag = f"unfreeze"

model_name = f"{model_str}_bs{str(batch_size)}_spe{str(steps_per_epoch)}_{fp_str}_{att_str}_{reg_str}_{foc_str}_{scl_str}_{aug_str}_{geo_mode}_{prepared_fname}_{ft_tag}{version}"
log_directory = os.path.join(saved_model_directory, f"log_{model_name}")

### Model testing ### commented out because this is for crown mapping?
#iou_thresh = 0.01 # IOU threshold for crown evaluation
#area_min_conf = 0.01
#external_pred_gpkg = 'path/to/predicted_crown_polygon.gpkg'


### Model inference ###
inf_model_name = "log_vgg_bs12_speNone_fp0d3_att0d1_reg0_foc1d5_scl0d5_aug_unit_prepared_1m_mixed_3sd_split1_frac1_BASE_scratch_v3" #changed from model_name
inference_mode = "dmp"   # options: 'dmp', 'amp', 'scale'
peak_method = "peak"  # options: 'peak', 'h_maxima', 'combine' | h_maxima and combine will dramatically increase computational time
min_distance = int(1/reproject_resolution)
peak_value = 0.35 # local_peak_maxima abs threshold if no params.ymal is provided
h_max_value = 0.15    # h_maxima prominence threshold

output_raster = True
output_centroids = True
output_crowns = False
inference_data_path = "/home/rxv7nr/us-tree-detection-geo-share/dl_project_input_data" ## CHANGED 
peak_value_str = n2t(peak_value)
inference_out_directory = os.path.join("/home/rxv7nr/us-tree-detection-geo-share/dl_project_output_data", f"inf_{model_name}/{os.path.basename(inference_data_path)}_{peak_value_str}")


