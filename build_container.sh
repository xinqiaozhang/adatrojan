sudo singularity build detector.simg detector.def


# singularity run \
# --bind /home/xinqiao/5-trojanai/round12-pdf/trojai-example \
# --nv \
# ./GBR_sub1.simg \
# configure \
# --scratch_dirpath=./scratch/ \
# --metaparameters_filepath=./metaparameters.json \
# --schema_filepath=./metaparameters_schema.json \
# --learned_parameters_dirpath=./new_learned_parameters/ \
# --configure_models_dirpath=/home/xinqiao/5-trojanai/round12-pdf/trojai-example/cyber-pdf-dec2022-train \
# --scale_parameters_filepath ./scale_params.npy


# singularity run \
# --bind /home/xinqiao/5-trojanai/round12-pdf/trojai-example \
# --nv \
# ./GBR_sub1.simg \
#  configure \
#  --scratch_dirpath ./scratch \
#  --metaparameters_filepath ./metaparameters.json \
#  --schema_filepath ./metaparameters_schema.json \
#  --learned_parameters_dirpath ./learned_parameters \
#  --configure_models_dirpath /home/xinqiao/5-trojanai/round12-pdf/trojai-example/cyber-pdf-dec2022-train/models \
#  --scale_parameters_filepath ./scale_params.npy

# singularity run \
# --bind /home/xinqiao/5-trojanai/round12-pdf/trojai-example \
# --nv \
# ./example_trojan_detector.simg \
# infer \
# --model_filepath=./model/id-00000002/model.pt \
# --result_filepath=./output.txt \
# --scratch_dirpath=./scratch/ \
# --examples_dirpath=./model/id-00000002/clean-example-data/ \
# --round_training_dataset_dirpath=/home/xinqiao/5-trojanai/round12-pdf/trojai-example/cyber-pdf-dec2022-train \
# --metaparameters_filepath=./metaparameters.json \
# --schema_filepath=./metaparameters_schema.json \
# --learned_parameters_dirpath=./new_learned_parameters/ \
# --scale_parameters_filepath ./scale_params.npy


# singularity run \
# --bind /home/xinqiao/5-trojanai/round12-pdf/trojai-example \
# --nv \
# ./GBR_sub1.simg \
# infer \
# --model_filepath=./model/id-00000002/model.pt \
# --result_filepath=./output.txt \
# --scratch_dirpath=./scratch/ \
# --examples_dirpath=./model/id-00000002/clean-example-data/ \
# --round_training_dataset_dirpath=/home/xinqiao/5-trojanai/round12-pdf/trojai-example/cyber-pdf-dec2022-train \
# --metaparameters_filepath=./metaparameters.json \
# --schema_filepath=./metaparameters_schema.json \
# --learned_parameters_dirpath=./learned_parameters/ \
# --scale_parameters_filepath ./scale_params.npy