def aggregate_array(in_arr, nagg_x=2, nagg_y=2):
    """


    :type in_arr: numpy.ndarray
    :type nagg_y: int
    """
    from skimage.util import view_as_blocks

    return view_as_blocks(in_arr, (nagg_x, nagg_y)).mean(axis=2).mean(axis=2)