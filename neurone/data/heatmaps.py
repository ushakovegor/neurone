import numpy as np

def define_borders(image_x, image_y, window_x, window_y,  kp_xs, kp_ys):
    x_l = kp_xs - (window_x - 1) / 2
    x_r = kp_xs + (window_x - 1) / 2
    y_l = kp_ys - (window_y - 1) / 2
    y_r = kp_ys + (window_y - 1) / 2
    
    x_min = np.max([x_l, np.zeros(len(kp_xs))], axis=0).astype(int)
    x_max = np.min([x_r, np.ones(len(kp_xs)) * image_x - 1 ], axis=0).astype(int) + 1
    y_min = np.max([y_l, np.zeros(len(kp_ys))], axis=0).astype(int)
    y_max = np.min([y_r, np.ones(len(kp_ys)) * image_y - 1 ], axis=0).astype(int) + 1
    
    window_x_min = np.where(x_l >= 0, 0, -x_l).astype(int)
    window_x_max = np.where(x_r < image_x, window_x, window_x - x_r + image_x - 1).astype(int)
    window_y_min = np.where(y_l >= 0, 0, -y_l).astype(int)
    window_y_max = np.where(y_r < image_y, window_y, window_y -  y_r + image_y -1).astype(int)
    
    return np.vstack([x_min, x_max, y_min, y_max]).T, np.vstack([window_x_min, window_x_max, window_y_min, window_y_max]).T  


def make_heatmap(x_size, y_size, keypoints, num_classes, base_bell):
    """
    Transforms keypoints to heatmap.

    Parameters
    ----------
    x_size: int
        x_size of the heatmap
    y_size: int
        y_size of heatmap
    keypoints: endoanalysis.targets.Keypoints
        keypoints to trasform
    sigma: int or ndarray of int
        standart deviation in pixels.
        If int, all keypoints will have the same sigmas,
        if ndarray, should have the shape (num_keypoits,)
    num_classes: int
        number of keypoints classes.

    Returns
    -------
    heatmap: ndarray
        heatmap for a given image. The shape is (num_classes, y_size, x_size)
    """

    bell_y, bell_x = base_bell.shape
    images_borders, windows_borders = define_borders(x_size, y_size, bell_x, bell_y, keypoints.x_coords(), keypoints.y_coords())

    heatmaps = np.zeros((num_classes, y_size, x_size))
    classes = keypoints.classes().astype(int)
    for image_borders, window_borders, class_i in zip(images_borders, windows_borders, classes):

        x_min, x_max, y_min, y_max = image_borders
        window_x_min, window_x_max, window_y_min, window_y_max = window_borders
        current_slice = heatmaps[class_i,y_min:y_max, x_min:x_max]
        bell = base_bell[window_y_min:window_y_max, window_x_min:window_x_max]

        heatmaps[class_i,y_min:y_max, x_min:x_max] = np.maximum(current_slice, bell)
        
    return heatmaps
