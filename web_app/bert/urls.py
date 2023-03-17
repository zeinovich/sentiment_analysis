from django.urls import re_path, include
from rest_framework.routers import DefaultRouter

from bert.views import MLRequestViewSet, PredictView


router = DefaultRouter(trailing_slash=False)
router.register(r"mlrequests", MLRequestViewSet, basename="mlrequests")
urlpatterns = [
    re_path(r"^prototype/", include(router.urls)),
    re_path(r"^prototype/predict$", PredictView.as_view(), name='predict')
]