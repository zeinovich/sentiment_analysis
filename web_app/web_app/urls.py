from django.contrib import admin
from django.urls import path
from bert.urls import urlpatterns as bert_urls

urlpatterns = [
    path("admin/", admin.site.urls),
]

urlpatterns += bert_urls