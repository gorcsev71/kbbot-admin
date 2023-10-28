from django.contrib import admin

from .models import Settings

# Register your models here.

@admin.register(Settings)
class SettingAdmin(admin.ModelAdmin):
    # list_display = [field.name for field in Settings._meta.get_fields()]

    def has_delete_permission(self, request, obj=None):
        return False

    def has_add_permission(self, request):
        return False

