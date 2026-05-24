from django.conf import settings
from django.shortcuts import redirect
from django.urls import reverse
from django.utils.http import urlencode

from .models import UserWebSettings


class ForcePasswordChangeMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        user = getattr(request, "user", None)
        if user and user.is_authenticated and self.should_enforce(request):
            web_settings, _ = UserWebSettings.objects.get_or_create(user=user)
            if web_settings.force_password_change:
                change_url = reverse("force-password-change")
                return redirect(f"{change_url}?{urlencode({'next': request.get_full_path()})}")
        return self.get_response(request)

    def should_enforce(self, request) -> bool:
        path = request.path
        allowed_exact = {
            reverse("force-password-change"),
            reverse("logout"),
        }
        if path in allowed_exact:
            return False
        if path.startswith("/api/"):
            return False
        static_url = getattr(settings, "STATIC_URL", "")
        media_url = getattr(settings, "MEDIA_URL", "")
        if static_url and path.startswith(static_url):
            return False
        if media_url and path.startswith(media_url):
            return False
        return True
