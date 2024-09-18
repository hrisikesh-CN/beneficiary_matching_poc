import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class Top80PercentEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column, top_percent=80, other_value=0):
        self.column = column
        self.top_percent = top_percent
        self.other_value = other_value
        self.encoding_map = {}
        self.decoding_map = {}
    
    def fit(self, X, y=None):
        category_counts = X[self.column].value_counts(ascending=False)
        total_count = category_counts.sum()
        
        # Start with cumulative count at 0
        cumulative_count = 0
        top_categories = []
        
        
        ### Accumulate Until 80%: As we iterate through the sorted categories, 
        ## we keep adding to cumulative_count until it reaches or exceeds 80% of total_count. 
        ##We stop adding more categories once the cumulative count surpasses the 80% threshold.
        
        for category, count in category_counts.items():
            cumulative_count += count
            if (cumulative_count / total_count)*100 <= self.top_percent: #
                top_categories.append(category)
            else:
                break
        
        # Create encoding dictionary for top categories and 'Other'
        self.encoding_map = {cat: i+1 for i, cat in enumerate(top_categories)}
        self.encoding_map.update({cat: self.other_value for cat in category_counts.index if cat not in top_categories})
        
        # Create reverse mapping for decoding
        self.decoding_map = {v: k for k, v in self.encoding_map.items()}
        self.decoding_map[self.other_value] = 'Other'
        
        return self

    def transform(self, X):
        # Map the column values to their respective encodings, unseen values mapped to 'Other'
        X_copy = X.copy()
        X_copy[self.column] = X_copy[self.column].apply(
            lambda x: self.encoding_map.get(x, self.other_value)  # Map unseen to 'Other'
        )
        return X_copy
    
    def inverse_transform(self, X):
        # Map encoded values back to original categories
        X_copy = X.copy()
        X_copy[self.column] = X_copy[self.column].map(self.decoding_map)
        return X_copy


class NameSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, full_name_column='Name', last_name_column='last_name'):
        self.full_name_column = full_name_column
        self.last_name_column = last_name_column
    
    def fit(self, X, y=None):
        # No fitting needed for this transformer
        return self
    
    def transform(self, X):
        # Ensure X is a DataFrame
        X = pd.DataFrame(X)
        
        # Check if the full_name_column exists in the DataFrame
        if self.full_name_column not in X.columns:
            raise ValueError(f"Column '{self.full_name_column}' not found in DataFrame")

        # Split the full names into first and last names
        X[self.last_name_column] = X[self.full_name_column].apply(
            lambda name: name.split()[-1] if isinstance(name, str) and len(name.split()) > 1 else name
        )
        X = X.drop(columns = [self.full_name_column])
        
        return X