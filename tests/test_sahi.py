from sahi import ImagePatch, split_image_in_windows
import numpy as np


   
   

def test_split_image():
    img = np.random.rand(800, 600, 3)
    patches = split_image_in_windows(
        img,
        patch_size=(100,100),
        overlapping_prportion=(0, 0)
    )
    assert len(patches) == 6*8

    patches = split_image_in_windows(
        img,
        patch_size=(100,100),
        overlapping_prportion=(0.5, 0.5)
    )
    assert len(patches) == 6*2*8*2

    patches_0_2 = split_image_in_windows(
        img,
        patch_size=(100,100),
        overlapping_prportion=(0.2, 0.2)
    )

    patches_0_7 = split_image_in_windows(
        img,
        patch_size=(100,100),
        overlapping_prportion=(0.7, 0.7)
    )
    assert len(patches_0_2) < len(patches_0_7)


    patches_0_75 = split_image_in_windows(
        img,
        patch_size=(100,100),
        overlapping_prportion=(0.75, 0.75)
    )
    assert len(patches_0_75) == 6*4*8*4


def test_bb_mapping():
    patch_size = (100, 80)
    patch_position_row = 29
    patch_position_col = 13

    target_image_size = (300, 300)
    predicted_bb = [6.0, 3.0, 27.0, 30.0]

    image = (np.random.rand(*patch_size, 3)*255).astype(np.uint8)

    image_patch = ImagePatch(
        img = image,
        starting_row=patch_position_row,
        starting_col=patch_position_col
    )
    image_patch.rescaling_preserving_aspect_ratio(target_size=target_image_size)
    assert image_patch.scaling_factor == 3
    assert image_patch.padding_rescaled_left == 30
    mapped_bbs = image_patch.map_bbs_to_original_image_coordinates(
        np.expand_dims(predicted_bb, 0)
    )

    mapped_bb = mapped_bbs[0]
    assert mapped_bb[0] == (6 - 30) / 3 + 13
    assert mapped_bb[1] == 3 / 3 + 29
    assert mapped_bb[2] == 27 / 3
    assert mapped_bb[3] == 30 / 3


