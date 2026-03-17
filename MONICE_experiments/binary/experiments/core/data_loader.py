import pickle

class TabularDataLoader:
    def __init__(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        self.X_explain = data['X_explain']
        self.y_explain = data['y_explain']
        self.feature_names = data['feature_names']
        self.categorical_indices = data['categorical_indices']
        self.continuous_indices = data['continuous_indices']
        
        # Load encoders with safety checks
        self.label_encoders = data.get('label_encoders', {})
        self.target_encoder = data.get('target_encoder', None)
        
        self.metadata = data['metadata']

        del data

    def decode_features(self, encoded_data, feature_names=None):
        """Decode encoded categorical features back to original values using encoder objects"""
        if not self.label_encoders:
            return encoded_data
            
        if feature_names is None:
            feature_names = self.feature_names
        
        decoded_data = encoded_data.copy()
        
        for feature_name, encoder in self.label_encoders.items():
            if feature_name in feature_names:
                idx = feature_names.index(feature_name)
                # Handle both 1D and 2D arrays
                if decoded_data.ndim == 1:
                    decoded_data[idx] = encoder.inverse_transform([int(decoded_data[idx])])[0]
                else:
                    decoded_data[:, idx] = encoder.inverse_transform(decoded_data[:, idx].astype(int))
        
        return decoded_data
    
    def decode_target(self, encoded_target):
        """Decode encoded target back to original values using encoder objects"""
        if self.target_encoder is None:
            return encoded_target
        
        if isinstance(encoded_target, (int, float)):
            return self.target_encoder.inverse_transform([int(encoded_target)])[0]
        else:
            return self.target_encoder.inverse_transform(encoded_target.astype(int))
    
    def has_encoders(self):
        """Check if encoders are available"""
        return bool(self.label_encoders) or self.target_encoder is not None
    
    
# USAGE
# fetcher = OpenMLFetcher('adult', 1590)
# fetcher.save()
# data = TabularDataLoader('./data/adult/data.pkl')
