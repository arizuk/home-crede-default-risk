def read_exclude_features():
    features = []
    with open("./exclude_features") as f:
        for line in f:
            feat = line.strip()
            if feat != '':
                features.append(feat)
    return features

def select_features(features):
    exclude_features = ['SK_ID_CURR']
    exclude_features = sum(list(map(lambda c: [c, f"{c}_x", f"{c}_y"], exclude_features)), [])
    # exclude_features = sum([exclude_features, read_exclude_features()], [])
    return [f for f in features if f not in exclude_features]