from metaflow import FlowSpec, step
import numpy as np

class TrainFlow(FlowSpec):

    @step
    def start(self):
        import pandas as pd
        import sklearn as sk
        from sklearn.model_selection import train_test_split

        df = pd.read_csv('../data/reference-data.csv')
        df = df[['animal-life-stage', 'animal-sex', 'animal-comments']]
        df['animal-life-stage'] = df['animal-life-stage'].str.split(' ').str[0]
        df['animal-sex'] = df['animal-sex'].apply(lambda x: 1 if x == 'm' else 0
                                                  if x == 'f' else np.NaN)
        df['prey'] = df['animal-comments'].str.split('prey_p_month: ').str[1]
        df = df.dropna()
        X = df[['animal-life-stage', 'animal-sex']]
        y = df['prey']
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(X, y)
        print("Data loaded successfully")
        self.next(self.train_linreg, self.train_dectree)

    @step
    def train_linreg(self):
        from sklearn.linear_model import LinearRegression

        self.model = LinearRegression()
        self.model.fit(self.train_data, self.train_labels)
        self.next(self.choose_model)

    @step
    def train_dectree(self):
        from sklearn.tree import DecisionTreeClassifier

        self.model = DecisionTreeClassifier(max_depth=5)
        self.model.fit(self.train_data, self.train_labels)
        self.next(self.choose_model)

    @step
    def choose_model(self, inputs):
        import mlflow
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        mlflow.set_experiment('metaflow-experiment')

        def score(inp):
            return inp.model, inp.model.score(inp.test_data, inp.test_labels)

        self.results = sorted(map(score, inputs), key=lambda x: -x[1])
        self.model = self.results[0][0]
        with mlflow.start_run():
            mlflow.sklearn.log_model(self.model, artifact_path = 'metaflow_train', registered_model_name="metaflow-wine-model")
            mlflow.end_run()
        self.next(self.end)

    @step
    def end(self):
        print('Scores:')
        print('\n'.join('%s %f' % res for res in self.results))
        print('Model:', self.model)


if __name__=='__main__':
    TrainFlow()
