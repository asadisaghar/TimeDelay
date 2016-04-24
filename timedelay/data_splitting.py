import numpy as np

def split_data_by_windows(data, test_fraction=0.33):
    # Split data to train and test sets by window_id
    # Cross validation is done inside LassoCV, so no need for a separate CV set
    windows = np.unique(data['window_id'])
    test_size = int(len(windows) * test_fraction)
    test_windows = windows[:test_size]
    train_windows = windows[test_size + 1:]
    test_end_ind = np.where(data['window_id'] == test_windows[-1])[0][-1]
    features_test = features[:test_end_ind]
    features_test = np.reshape(features_test, (len(features_test), 1))
    labels_test = labels[:test_end_ind]
    features_train = features[test_end_ind + 1:]
    features_train = np.reshape(features_train, (len(features_train), 1))
    labels_train = labels[test_end_ind + 1:]
    return features_train, labels_train, features_test, labels_test
