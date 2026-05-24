from django.apps import AppConfig


class MeetingsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "meetings"

    def ready(self):
        from django.conf import settings
        from django.db.models.signals import post_save

        from .models import UserWebSettings

        def create_user_web_settings(sender, instance, created, **kwargs):
            if created:
                UserWebSettings.objects.get_or_create(user=instance)

        post_save.connect(
            create_user_web_settings,
            sender=settings.AUTH_USER_MODEL,
            dispatch_uid="meetings.create_user_web_settings",
        )
