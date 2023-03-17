from rest_framework import serializers
from bert.models import MLRequest

class MLRequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = MLRequest
        read_only_fields = (
            "id",
            "model_name",
            "input_data",
            "full_response",
            "response",
            "created_at"
        )
        fields =  (
            "id",
            "model_name",
            "input_data",
            "full_response",
            "response",
            "feedback",
            "created_at"
        )

