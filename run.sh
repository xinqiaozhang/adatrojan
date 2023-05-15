python entrypoint.py infer \
--model_filepath /data/xinqiao/round13/object-detection-feb2023-example/model.pt \
--result_filepath ./output.txt \
--scratch_dirpath ./scratch \
--examples_dirpath /data/xinqiao/round13/object-detection-feb2023-example/clean-example-data \
--round_training_dataset_dirpath /data/xinqiao/round13/object-detection-feb2023-example/ \
--learned_parameters_dirpath ./learned_parameters \
--metaparameters_filepath ./metaparameters.json \
--schema_filepath=./metaparameters_schema.json