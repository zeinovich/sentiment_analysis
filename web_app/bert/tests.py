from django.test import TestCase

from bert.bert_model import BERT


class BERT_Test(TestCase):
    def test_bert(self):
        input_data = {"text": "I hate this movie"}
        error_data = 23

        bert = BERT()
        response = bert.make_prediction(input_data)
        error_response = bert.make_prediction(error_data)

        self.assertEqual("OK", response["status"])
        self.assertTrue("rating" in response)
        self.assertEqual(1, response["rating"])

        self.assertEqual("Error", error_response["status"])
        self.assertTrue("rating" not in error_response)
