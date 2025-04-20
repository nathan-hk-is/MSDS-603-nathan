from metaflow import FlowSpec, step, Flow
import numpy as np
import mlflow

class ScoreFlow(FlowSpec):

    @step
    def start(self):
        import pandas as pd
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
        self.next(self.load_model)
    
    @step
    def load_model(self):
        run = Flow('TrainFlow').latest_run
        self.model = run['end'].task.data.model
        self.next(self.predict)

    @step
    def predict(self):
        import pandas as pd
        self.preds = self.model.predict(self.test_data)
        out = pd.DataFrame({'prediction': self.preds, 'actual': self.test_labels})
        out.to_csv('preds.csv', index=False)
        self.next(self.end)

    @step
    def end(self):
        print('First five predictions:')
        print(self.preds[:5])


if __name__=='__main__':
    ScoreFlow()
