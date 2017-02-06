def aggregate_array(in_arr, nagg_x=2, nagg_y=2):
    """


    :type in_arr: numpy.ndarray
    :type nagg_y: int
    """
    from skimage.util import view_as_blocks

    if nagg_x == 1 and nagg_y == 1:
        return in_arr

    print(in_arr.shape, nagg_x, nagg_y)

    assert in_arr.shape[0] % nagg_x == 0
    assert in_arr.shape[1] % nagg_y == 0



    return view_as_blocks(in_arr, (nagg_x, nagg_y)).mean(axis=2).mean(axis=2)