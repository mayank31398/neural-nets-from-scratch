import numpy as np

def Accuracy(predictions, y):
    predictions = predictions.reshape(predictions.shape[0], 1)
    y = y.reshape(y.shape[0], 1)
    return (predictions == y).sum() / y.shape[0]

def Precision(predictions, y):
    # This is correctly detected divided by all detected
    classes = list(set(y))

    predictions = predictions.reshape(predictions.shape[0], 1)
    y = y.reshape(y.shape[0], 1)

    precision = {}
    for i in classes:
        temp = predictions == i
        TP = (y[temp] == i).sum(axis = 0)
        precision[i] = TP / temp.sum(axis = 0)
    
    return precision

def Recall(predictions, y):
    # This is correctly detected divided by total positive samples in the dataset
    classes = list(set(y))

    predictions = predictions.reshape(predictions.shape[0], 1)
    y = y.reshape(y.shape[0], 1)

    recall = {}
    for i in classes:
        temp = y == i
        TP = (predictions[temp] == i).sum(axis = 0)
        recall[i] = TP / temp.sum(axis = 0)
    
    return recall

def F1score(predictions, y):
    classes = list(set(y))

    f1_score = {}
    recall = Recall(predictions, y)
    precision = Precision(predictions, y)
    for i in classes:
        f1_score[i] = 2 * recall[i] * precision[i] / (recall[i] + precision[i])
    
    return f1_score

def roc_curve(y_true, y_score, pos_label=None, sample_weight=None,
              drop_intermediate=True):
    fps, tps, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
    
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True,
                                      np.logical_or(np.diff(fps, 2),
                                                    np.diff(tps, 2)),
                                      True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    if tps.size == 0 or fps[0] != 0:
        # Add an extra threshold position if necessary
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return fpr, tpr, thresholds

def _binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    # ensure binary classification if pos_label is not specified
    if pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        fps = np.cumsum(weight)[threshold_idxs] - tps
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]

def column_or_1d(y):
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        return np.ravel(y)