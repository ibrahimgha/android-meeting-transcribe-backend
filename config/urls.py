from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.shortcuts import redirect
from django.urls import include, path

from meetings.auth_views import FirstLoginLoginView, ForcePasswordChangeView

urlpatterns = [
    path("", lambda request: redirect("web-meetings"), name="home"),
    path("admin/", admin.site.urls),
    path("accounts/login/", FirstLoginLoginView.as_view(), name="login"),
    path("accounts/force-password-change/", ForcePasswordChangeView.as_view(), name="force-password-change"),
    path("accounts/", include("django.contrib.auth.urls")),
    path("meetings/", include("meetings.web_urls")),
    path("api/", include("meetings.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
