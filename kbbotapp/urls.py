from django.urls import path

from .views import Home, Chat
from .views import ConfigHome, SettingsHome, SettingsUpdate,SettingsReset
from .views import IndexHome, IndexCreateJSON, IndexUpload, IndexCreate
from .views import DocumentList, DocumentAdd, DocumentDetail, DocumentDelete, DocumentUpload

app_name = "kbbotapp"

urlpatterns = [
    path("", Home.as_view(), name="home"),
    path("chat/", Chat.as_view(), name="chat"),
    path("config/", ConfigHome.as_view(), name="config_home"),
    path("config/settings/", SettingsHome.as_view(), name="config_settings_home"),
    path("config/settings/<pk>/", SettingsUpdate.as_view(), name="config_settings_update"),
    path("config/settings/<pk>/reset/", SettingsReset.as_view(), name="config_settings_reset"),
    path("config/index/", IndexHome.as_view(), name="config_index_home"),
    path("config/index/create/", IndexCreate.as_view(), name="config_index_create"),

    path("config/index/refresh/", IndexCreateJSON.as_view(), name="config_index_refresh"),
    path("config/index/upload/", IndexUpload.as_view(), name="config_index_upload"),
    path("config/documents/", DocumentList.as_view(), name="config_document_list"),
    path("config/documents/upload/", DocumentUpload.as_view(), name="config_document_upload"),
    path("config/document/add/", DocumentAdd.as_view(), name="config_document_add"),
    path("config/document/<pk>/", DocumentDetail.as_view(), name="config_document_detail"),
    path("config/document/<pk>/delete/", DocumentDelete.as_view(), name="config_document_delete"),

]