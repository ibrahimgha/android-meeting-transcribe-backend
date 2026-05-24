from django.contrib import messages
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.views import LoginView
from django.shortcuts import redirect
from django.urls import reverse
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme, urlencode
from django.views.generic.edit import FormView

from .models import UserWebSettings


class FirstLoginLoginView(LoginView):
    template_name = "registration/login.html"
    redirect_authenticated_user = True

    def get_success_url(self):
        settings, _ = UserWebSettings.objects.get_or_create(user=self.request.user)
        if settings.force_password_change:
            next_url = self.get_redirect_url() or reverse("web-meetings")
            return f"{reverse('force-password-change')}?{urlencode({'next': next_url})}"
        return super().get_success_url()


class ForcePasswordChangeView(FormView):
    form_class = PasswordChangeForm
    template_name = "registration/force_password_change.html"

    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect("login")
        settings, _ = UserWebSettings.objects.get_or_create(user=request.user)
        if not settings.force_password_change:
            return redirect(self.get_safe_next_url())
        return super().dispatch(request, *args, **kwargs)

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs["user"] = self.request.user
        return kwargs

    def form_valid(self, form):
        user = form.save()
        update_session_auth_hash(self.request, user)
        settings, _ = UserWebSettings.objects.get_or_create(user=user)
        settings.force_password_change = False
        settings.password_changed_at = timezone.now()
        settings.save(update_fields=["force_password_change", "password_changed_at", "updated_at"])
        messages.success(self.request, "Password changed.")
        return redirect(self.get_safe_next_url())

    def get_safe_next_url(self):
        next_url = self.request.GET.get("next") or self.request.POST.get("next") or reverse("web-meetings")
        if url_has_allowed_host_and_scheme(
            next_url,
            allowed_hosts={self.request.get_host()},
            require_https=self.request.is_secure(),
        ):
            return next_url
        return reverse("web-meetings")
