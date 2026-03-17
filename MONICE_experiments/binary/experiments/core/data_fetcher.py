import os
import pickle

import numpy as np
import pandas as pd
import openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class OpenMLFetcher:
    def __init__(self, dataset_name, dataset_id, test_size=0.2, explain_sample_size=200):
        self.dataset_name = dataset_name
        self.dataset_id = dataset_id
        self.folder_path = f'./data/{dataset_name}'
        
        # Create folders if not exist
        if not os.path.isdir(self.folder_path):
            os.makedirs(os.path.join(self.folder_path, 'results'))
        
        # Fetch dataset from OpenML
        self.dataset = openml.datasets.get_dataset(dataset_id)
        self.target_name = self.dataset.default_target_attribute
        # Get data with metadata
        X, y, _, attribute_names = self.dataset.get_data(target=self.target_name)
        self.attribute_names = list(attribute_names)
        
        # dataframe
        dataframe = pd.DataFrame(X, columns=self.attribute_names)
        dataframe[self.target_name] = y
        
        # Clean data
        dataframe = dataframe.dropna()
        dataframe = dataframe.drop_duplicates()

        # First, identify features to remove (string with many values, date features)
        features_to_remove = self._identify_features_to_remove(dataframe)
        
        # Remove unwanted features
        if features_to_remove:
            print(f"Removing features: {features_to_remove}")
            dataframe = dataframe.drop(columns=features_to_remove)
            self.attribute_names = [name for name in self.attribute_names if name not in features_to_remove]
        
        feature_types, categorical_indices, continuous_indices = self._get_feature_types()
        
        # Process features and target
        features_df = dataframe[self.attribute_names].copy()  # Feature columns
        labels = dataframe[self.target_name].values  # Target column
        
        # label encoder  for cat feats
        self.label_encoders = {}
        self.target_encoder = None
        
        for idx in categorical_indices:
            feature_name = self.attribute_names[idx]
            # if not self._is_already_encoded(features_df[feature_name].values):
            encoder = LabelEncoder()
            features_df[feature_name] = encoder.fit_transform(features_df[feature_name].astype(str))
            self.label_encoders[feature_name] = encoder
        
        # label for target
        self.target_encoder = LabelEncoder()
        labels = self.target_encoder.fit_transform(labels.astype(str))
        # store in numpy
        features = features_df.values
        
        # Split 
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, stratify=labels, test_size=test_size, random_state=42
        )
        
        # Extract explain subset from test set
        # If test set is large enough, sample explain_sample_size
        # If test set is smaller, use the entire test set
        if X_test.shape[0] > explain_sample_size:
            
            explain_ratio = explain_sample_size / X_test.shape[0]
            print(f"Explain ratio: {explain_ratio}")
            _, X_explain, _, y_explain = train_test_split(
                X_test, y_test, stratify=y_test, 
                test_size=explain_ratio, random_state=42
            )
        else:
            X_explain = X_test.copy()
            y_explain = y_test.copy()
            print(f"Warning: Test set ({X_test.shape[0]}) smaller than desired explain size ({explain_sample_size}). Using entire test set.")
        
        print(f"{self.dataset_name} | explain set: {X_explain.shape[0]} | test set: {X_test.shape[0]}")
        
        # Save dataset
        self.dataset_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_explain': X_explain,
            'y_explain': y_explain,
            'feature_names': self.attribute_names,
            'categorical_indices': categorical_indices,
            'continuous_indices': continuous_indices,
            'label_encoders': self.label_encoders,
            'target_encoder': self.target_encoder,
            'metadata': self._prepare_metadata(feature_types)
        }
        
        # Create description file
        self._create_description_file(feature_types, features_to_remove)
    
    def _identify_features_to_remove(self, dataframe):
        """Identify features to remove based on OpenML metadata"""
        features_info = self.dataset.features # dict: index, name, data_type
        features_to_remove = []
        
        for feature_name in self.attribute_names:
            feature_info = None
            for feat_id, feat in features_info.items():
                if feat.name == feature_name:
                    feature_info = feat
                    break
            
            if feature_info:
                data_type = feature_info.data_type.lower()
                
                if data_type == 'string':
                    # Check distinct values for string features
                    distinct_count = len(pd.Series(dataframe[feature_name]).dropna().unique())
                    
                    if distinct_count < 100:
                        continue
                    else:
                        features_to_remove.append(feature_name)
                elif data_type == 'date':
                    features_to_remove.append(feature_name)
                
        return features_to_remove
    
    def _get_feature_types(self):
        """Get feature types from OpenML dataset metadata for remaining features"""
        features_info = self.dataset.features
        feature_types = {}
        categorical_indices = []
        continuous_indices = []
        
        for idx, feature_name in enumerate(self.attribute_names):
            # Find the feature info by name
            feature_info = None
            for feat_id, feat in features_info.items():
                if feat.name == feature_name:
                    feature_info = feat
                    break
            
            if feature_info:
                data_type = feature_info.data_type.lower()
                
                if data_type in ['nominal', 'string', 'categorical']:  
                    feature_types[feature_name] = {
                        'type': 'categorical',
                        'index': idx
                    }
                    categorical_indices.append(idx)
                elif data_type in ['numeric']: 
                    feature_types[feature_name] = {
                        'type': 'numeric',
                        'index': idx
                    }
                    continuous_indices.append(idx)
                else:
                    raise ValueError(f"Unknown data type: {data_type}")

        return feature_types, categorical_indices, continuous_indices
    
    # def _is_already_encoded(self, values):
    #     """Check if categorical values are already encoded as integers"""
    #     try:
    #         # Check if all values are numeric and integers
    #         numeric_values = pd.to_numeric(values, errors='coerce')
    #         return not numeric_values.isna().any() and all(float(val).is_integer() for val in numeric_values if not pd.isna(val))
    #     except:
    #         return False
    
    def _prepare_metadata(self, feature_types):
        """Prepare metadata including encoders for future use"""
        metadata = {
            'dataset_id': self.dataset_id,
            'dataset_name': self.dataset_name,
            'target_attribute': self.dataset.default_target_attribute,
            'feature_types': feature_types,
            'label_encoders': {},
            'target_encoder': None
        }
        
        for feature_name, encoder in self.label_encoders.items():
            metadata['label_encoders'][feature_name] = {
                'values_mappings': [(cls, idx) for idx, cls in enumerate(encoder.classes_)]
            }
        
        if self.target_encoder:
            metadata['target_encoder'] = {
                'values_mappings': [(cls, idx) for idx, cls in enumerate(self.target_encoder.classes_)]
            }
        
        return metadata
    
    def _create_description_file(self, feature_types, features_to_remove):
        """Create description.md file with dataset information"""
        description_path = os.path.join(self.folder_path, 'description.md')
        
        with open(description_path, 'w', encoding='utf-8') as f:
            f.write(f"# Dataset: {self.dataset_name}\n\n")
            f.write(f"**OpenML ID**: {self.dataset_id}\n\n")
            f.write(f"**Target Attribute**: {self.dataset.default_target_attribute}\n\n")
            
            f.write("## Description\n\n")
            if self.dataset.description:
                f.write(f"{self.dataset.description}\n\n")
            
            # Access features directly from dataset object
            features_info = self.dataset.features
            
            f.write("## Features\n\n")
            for feat_id, feat in features_info.items():
                f.write(f"- **{feat.name}** ({feat.data_type})\n")

            f.write(f"\n## Features to remove\n\n")
            for feature in features_to_remove:
                f.write(f"- {feature}\n")
                
            f.write(f"\n## Feature types after preprocessing\n\n")
            for feature, feat_type in feature_types.items():
                f.write(f"- {feature}: {feat_type['type']}\n")
                
            f.write(f"\n## Feature Encodings\n\n")
            for feature_name, encoder in self.label_encoders.items():
                f.write(f"### {feature_name}\n")
                f.write("| Original Value | Encoded Value |\n")
                f.write("| --- | --- |\n")
                for idx, cls in enumerate(encoder.classes_):
                    f.write(f"| {cls} | {idx} |\n")
                f.write("\n")
            
            f.write(f"\n## Class Mapping\n\n")
            if self.target_encoder:
                f.write("| Original Class | Encoded Value |\n")
                f.write("| --- | --- |\n")
                for idx, cls in enumerate(self.target_encoder.classes_):
                    f.write(f"| {cls} | {idx} |\n")
            else:
                f.write("No encoding was applied to target values (already numeric).\n")
            
            f.write(f"\n## Dataset Statistics\n\n")
            qualities = self.dataset.qualities
            if qualities:
                f.write(f"- Number of instances: {qualities.get('NumberOfInstances', 'N/A')}\n")
                f.write(f"- Number of features: {qualities.get('NumberOfFeatures', 'N/A')}\n")
                f.write(f"- Number of classes: {qualities.get('NumberOfClasses', 'N/A')}\n")
                f.write(f"- Missing values: {qualities.get('NumberOfMissingValues', 'N/A')}\n")
    
    def save(self):
        """Save processed dataset to disk as pickle."""
        save_path = os.path.join(self.folder_path, 'data.pkl')
        with open(save_path, 'wb') as file:
            pickle.dump(self.dataset_data, file)
