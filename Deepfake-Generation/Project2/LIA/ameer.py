from datapipes.base_pipeline import BasePipelineCreator
from datapipes.decoders.image_decoder import ImageDecoder, VideoDecoder
from datapipes.decoder import Decoder
import json

def read_tar(dataset):

    TAR_DATASET = dataset

    factory = BasePipelineCreator(TAR_DATASET)
    # Read the JSON file
    with open(TAR_DATASET+'/'+'metadata.json', 'r') as file:
        data = json.load(file)
        sample_count_train = data['train']['sample_count']
        sample_count_val = data['val']['sample_count']

    for subset in factory.get_subsets():
        print(f"Subset: {subset}")
        tars = factory.get_tar_files_for_subsets(subsets=[subset])
        print(f"Num Shards: {len(tars)}")
        groups = factory.get_component_groups(subset)
        print(f"Num Slices: {len(tars) / len(groups)}")
        shardlen = factory.get_average_shard_sample_count(subset)
        print(f"Avg Shard Len: {shardlen}")
        stats = factory.get_component_groups_stats(subset)
        for group in groups:
            if subset == "train":
                print(f"  Component Group: {group} {stats[group]}")
    components_train = [x for group, stats in factory.get_component_groups_stats("train").items() if group != "radar" for x in stats["all_components"]]
    for subset in factory.get_subsets():
        print(f"Subset: {subset}")
        tars = factory.get_tar_files_for_subsets(subsets=[subset])
        print(f"Num Shards: {len(tars)}")
        groups = factory.get_component_groups(subset)
        print(f"Num Slices: {len(tars) / len(groups)}")
        shardlen = factory.get_average_shard_sample_count(subset)
        print(f"Avg Shard Len: {shardlen}")
        stats = factory.get_component_groups_stats(subset)
        for group in groups:
            if subset == "val":
                print(f"  Component Group: {group} {stats[group]}")
    components_val = [x for group, stats in factory.get_component_groups_stats("val").items() if group != "radar" for x in stats["all_components"]]

    pipe_train = factory.create_datapipe(
        subsets=["train"], 
        shuffle_buffer_size=2048,
        shuffle_shards=True,
        components=components_train)
    
    pipe_train = pipe_train.map(fn=Decoder({'rgb': VideoDecoder()}))

    pipe_val = factory.create_datapipe(
        subsets=["val"], 
        shuffle_buffer_size=2048,
        shuffle_shards=True,
        components=components_val)
    
    pipe_val = pipe_val.map(fn=Decoder({'rgb': VideoDecoder()}))
    
    return pipe_train,pipe_val,sample_count_train,sample_count_val