from rest_framework import viewsets
from rest_framework import mixins
from rest_framework import views, status
from rest_framework.response import Response

import json
from bert import bert_predictor
from bert.models import MLRequest
from bert.serializers import MLRequestSerializer

# Create your views here.
class MLRequestViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet,
    mixins.UpdateModelMixin
):
    serializer_class = MLRequestSerializer
    queryset = MLRequest.objects.all()

class PredictView(views.APIView):
    def post(self, request):
        prediction = bert_predictor.make_prediction(request.data)
        rating = prediction['rating'] if 'rating' in prediction else 'error'

        ml_request = MLRequest(
            model_name=bert_predictor.name,
            input_data=json.dumps(request.data),
            full_response=prediction,
            response=rating,
            feedback=""
        )
        ml_request.save()
        prediction['request_id'] = ml_request.id

        return Response(prediction)