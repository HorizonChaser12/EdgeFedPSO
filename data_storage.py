import pickle


def save_data(data, filename):
    """Saves data to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_data(filename):
    """Loads data from a pickle file."""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
