import torch
from transformers import BertTokenizer
import logging

LOGGER = logging.getLogger(__name__)


class BERT:
    def __init__(self) -> None:
        '''
        ## BERT

        -------------------------

        Creates BERT model object

        -------------------------
        ### Attributes:
            device: The device to run the model on.
            model: The BERT model.
            tokenizer: The BERT tokenizer.
            rating_dict: The dictionary to convert BERT prediction to rating.

        -------------------------
        ### Methods:
            __name__: Returns the name of the model.
            encode: Encodes the input data to BERT format.
            predict: Predicts the rating of the input data.
            make_response: Makes the response of the prediction.
            make_prediction: Makes the prediction of the input data.
        '''

        MODEL_PATH = "bert/ml_models/bert_trained.pt"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(MODEL_PATH, map_location=self.device)
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.rating_dict = {0: 1, 1: 2, 2: 3, 3: 4, 4: 7, 5: 8, 6: 9, 7: 10}
        self.model.eval()
        LOGGER.info("BERT initialized")

    @property
    def name(self):
        return 'BERT_v0.0.1'
    
    def encode(self, input_data: dict) -> tuple[torch.Tensor]:
        '''
      
        Encodes the input data to BERT format
        
        -------------------------
        ### Arguments:
            input_data: dict or str
                The input data to encode.

        ### Returns:
            input_ids: torch.Tensor
                The input ids of the input data.
            input_mask: torch.Tensor 
                The input mask of the input data.
        '''

        if isinstance(input_data, dict):
            review = input_data["text"]

        elif isinstance(input_data, str):
            review = input_data

        else:
            LOGGER.error("Bad input")
            raise TypeError("BERT got bad input")

        encoded_dict = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        return encoded_dict["input_ids"], encoded_dict["attention_mask"]

    def predict(self, input_ids: torch.Tensor, input_mask: torch.Tensor) -> tuple[tuple, float]:
        '''
        Predicts the rating of the input data.
        
        -------------------------
        ### Arguments:
            input_ids: torch.Tensor
                The input ids of the input data.
                input_mask: torch.Tensor
                The input mask of the input data.
        
        ### Returns:
            prediction: tuple
                The probability and the prediction of the input data.'''
        
        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)

        output = self.model(
            input_ids,
            token_type_ids=None,
            attention_mask=input_mask,
        )

        logits = output.logits
        log_soft = torch.softmax(logits, 1).detach().cpu()
        proba, pred = torch.max(log_soft, 1)
        proba, pred = proba.numpy()[0], pred.numpy()[0]

        weighted = sum(i * x for i, x in enumerate(log_soft.numpy()[0]))
        weighted = weighted + 3 if weighted > 4 else weighted # reverse convertion
        return (proba, pred), weighted

    def make_response(self, prediction: tuple[float]) -> dict:
        '''
        Makes the response of the prediction.

        -------------------------   
        ### Arguments:
            prediction: tuple
                The probability and the prediction of the input data.
            
        ### Returns:
            response: dict
                The response of the prediction.
        '''

        rating = self.rating_dict[prediction[1]]

        return {
            "probability": prediction[0],
            "rating": rating,
            "sentiment": "positive" if rating > 5 else "negative",
            "status": "OK",
        }

    def make_prediction(self, input_data: dict) -> dict:
        '''
        Makes the prediction of the input data.
        
        -------------------------
        ### Arguments:
            input_data: dict or str
                The input data to encode.
                
        ### Returns:
            prediction: dict
                The response of the prediction.'''
        
        try:
            ids_mask = self.encode(input_data)
            prediction, weighted = self.predict(*ids_mask)
            prediction = self.make_response(prediction)
            LOGGER.info(f"{weighted=}   {prediction['rating']=}")
            return prediction

        except Exception as e:
            LOGGER.error(f"Error while predicting\n{e}")
            return {"status": "Error", "message": e}
