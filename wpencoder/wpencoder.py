import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

class Encoder:
    """
    A class used to convert an id-occurrence dataframe to an
    id-count aggregated dataframe. By default, counts are scaled and
    PCA vectorized.

    ...

    Attributes
    ----------
    df : pandas.DataFrame
        a dataframe containing id-occurrence data
    target_column : str
        the name of the column in df to count
    id_column : str
        the name of the column in df to aggregate up to
    ids : list
        optional. a list of ids to include in the index of the aggregated df
    oh_encoder: sklearn.preprocessing.OneHotEncoder
        a OneHotEncoder
    scaler: sklearn.preprocessing.MinMaxScaler or StandardScaler
        a scaler, either MinMax or Standard
    pca: sklearn.decomposition.PCA
        a PCA vectorizer

    Methods
    -------
    series_to_array(target)
        Convert Pandas series to Numpy array
    fill_ids(df=None)
        Ensure that the dataframe has all of the ids in the class' ids list
    oh_encoder_fit()
        Fits the one-hot encoder using the class' dataframe
    oh_encoder_transform(target)
        Transforms a new array using the class' fitted one-hot encoder
    scaler_fit()
        Fits the scaler using the class' dataframe
    scaler_transform(summarized_df)
        Transforms a new dataframe using the class' fitted scaler
    pca_fit()
        Fits the PCA transformer using the class' dataframe
    pca_transform(summarized_df)
        Transforms a new dataframe using the class' fitted PCA transformer
    encode_df(df=None)
        Helper method to one hot encode the target column
    encode_combine_df(df=None)
        One hot encode the target column then combine with the dataframe
    summarize_df(df=None, prefix=None, scaled=True, pca=True):
        Convert an id-occurrence dataframe to an id-count aggregated dataframe
    """

    def __init__(self, df, target_column, id_column, ids=None, scaler='MinMax'):
        # Name the class object based on the provided dataframe
        try:
            self.__name__ = df.__name__
        except AttributeError:
            self.__name__ = [x for x in globals() if globals()[x] is df][0]

        # Store the provided dataframe, target, and id columns
        self.df = df
        self.target_column = target_column
        self.id_column = id_column

        # List of ids to include in the index
        self.ids = ids

        # One hot encoder
        self.oh_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

        # Scaler
        if scaler == 'MinMax':
            self.scaler = MinMaxScaler()
        elif scaler == 'Standard':
            self.scaler = StandardScaler()

        # PCA set to explain 95% of variance by default
        self.pca = PCA(n_components = 0.95)

        # Fit the transformers using the class dataframe
        self.oh_encoder_fit()
        self.scaler_fit()
        self.pca_fit()

    @staticmethod
    def series_to_array(target):
        """Convert Pandas series to Numpy array

        Keyword arguments:
            target -- either a Pandas series or a Numpy array

        Returns:
            a Numpy array
        """

        # Check if the provided target is a Pandas series
        is_series = type(target) == pd.core.series.Series

        return target.to_numpy().reshape(-1, 1) if is_series else target

    def fill_ids(self, df=None):
        """Ensure that the dataframe has all of the ids in the class' ids list
            intended for joining multiple encoded dataframes

        Keyword arguments:
            df -- the dataframe to check.
                Index should conceptually match self.ids

        Returns:
            df reindexed to have all of the ids in the list
        """

        # If no dataframe provided use the class dataframe
        df = self.df.copy() if df is None else df.copy()

        # Reindex the dataframe using the id list
        if self.ids:
            df = df.reindex(self.ids).fillna(0)

        return df

    def oh_encoder_fit(self):
        """Fits the one-hot encoder using the class' dataframe
        """

        # Get the class dataframe's target column as a numpy array
        target = self.series_to_array(self.df[self.target_column])

        # Fit the one hot encoder
        self.oh_encoder.fit(target)

    def oh_encoder_transform(self, target):
        """Transforms a new array using the class' fitted one-hot encoder

        Keyword arguments:
            target -- a Numpy array or Pandas series to transform

        Returns:
            a Pandas dataframe with transformed values
        """

        # If the one hot encoder is not fit, fit now
        if self.oh_encoder is None:
            self.oh_encoder_fit()

        # Get the fitted one hot encoder categories
        columns=self.oh_encoder.categories_[0]

        # Transform the target using the fitted one hot encoder
        transformed = pd.DataFrame(
            self.oh_encoder.transform(self.series_to_array(target)),
            columns=columns
        )

        return transformed

    def scaler_fit(self):
        """Fits the scaler using the class' dataframe
        """

        # Summarize the data frame (one-hot encode and sum up to id col)
        summarized = self.summarize_df(self.df, scaled=False, pca=False)

        # Fit the scaler
        self.scaler.fit(summarized)

    def scaler_transform(self, summarized_df):
        """Transforms a new dataframe using the class' fitted scaler

        Keyword arguments:
            summarized_df -- a Pandas dataframe to scale.
                Possibly from encode_combine_df

        Returns:
            a Pandas dataframe with scaled values
        """

        # If the scaler is not fit, fit now
        if self.scaler is None:
            self.scaler_fit()

        # Transform the dataframe using the scaler
        transformed = pd.DataFrame(
            self.scaler.transform(summarized_df),
            index=summarized_df.index,
            columns=summarized_df.columns
        )

        return transformed

    def pca_fit(self):
        """Fits the PCA transformer using the class' dataframe
        """

        # Summarize the data frame (one-hot encode and sum up to id col)
        summarized = self.summarize_df(self.df, scaled=False, pca=False)

        # Fit the PCA vectorizer
        self.pca.fit(summarized.values)

    def pca_transform(self, summarized_df):
        """Transforms a new dataframe using the class' fitted PCA transformer

        Keyword arguments:
            summarized_df -- a Pandas dataframe to scale.
                Possibly from encode_combine_df

        Returns:
            a Pandas dataframe with scaled values
        """

        # If the PCA is not fit, fit now
        if self.pca is None:
            self.pca_fit()

        # Transform the dataframe using PCA transformer
        transformed = pd.DataFrame(
            self.pca.transform(summarized_df),
            index=summarized_df.index
        )

        return transformed

    def encode_df(self, df=None):
        """Helper method to one hot encode the target column

        Keyword arguments:
            df -- the dataframe to transform.
                Index should conceptually match self.ids

        Returns:
            a Pandas dataframe with an incremented index and
                columns from the encoder categories
        """

        # If no dataframe provided use the class dataframe
        df = self.df.copy() if df is None else df.copy()

        # Get the target column from the dataframe
        target = self.series_to_array(df[self.target_column])

        # Transform the target column using the class encoder
        return self.oh_encoder_transform(target)

    def encode_combine_df(self, df=None):
        """One hot encode the target column then combine with the dataframe

        Keyword arguments:
            df -- the dataframe to transform.
                Index should conceptually match self.ids

        Returns:
            a Pandas dataframe with the id_column index and
                columns from the original df and encoder categories
        """

        # If no dataframe provided use the class dataframe
        df = self.df.copy() if df is None else df.copy()

        # One hot encode the dataframe's target column
        encoded = self.encode_df(df)

        # Drop the target column
        df = df.drop(columns=self.target_column)

        # Recombine the encoded columns with the provided df
        combined = pd.concat([df, encoded], axis=1)

        return combined

    def summarize_df(self, df=None, prefix=None, scaled=True, pca=True):
        """Summarize the input dataframe, which is assumed to be long
            and contain one row per occurrence along the id_column.
            The output will be one-hot encoded first, then aggregated
            and summed at the id_column grain.

        Keyword arguments:
            df -- the dataframe to transform. both id_column and
                target_column should appear in the df's columns.
                if None (default) the class df will be used.

            prefix -- a prefix string to prepend to output column names.
                if None (default) the class __name__ will be used.

            scaled -- should the output columns be scaled using the scaler?
                defaults to True

            pca -- should PCA be applied to the output frame?

        Returns:
            a Pandas dataframe with the id_column index and
                columns from the encoder categories
        """

        # If no dataframe provided use the class dataframe
        df = self.df.copy() if df is None else df.copy()

        # Encode the dataframe
        encoded = self.encode_combine_df(df[[self.id_column, self.target_column]])

        # Fill in any missing ids from the id list
        encoded = self.fill_ids(encoded.groupby(self.id_column).agg(sum))

        # Scale
        if scaled:
            encoded = self.scaler_transform(encoded)

        # Apply PCA
        if pca:
            encoded = self.pca_transform(encoded)

        # Prepend the prefix string to the column names
        prefix = prefix if prefix else self.__name__
        encoded.columns = [prefix + '_' + str(c) for c in encoded.columns.tolist()]

        return encoded
