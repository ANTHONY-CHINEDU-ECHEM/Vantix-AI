"""Access control evaluation.

Vantix never decides who belongs to a project or group; the host platform does, and passes
that knowledge in the ``Principal`` with every request. Vantix only evaluates item ACLs
against that principal, both inside the index query (as a filter) and again in memory as a
defence in depth step before anything leaves the service.
"""

from __future__ import annotations

from vantix.domain.models import Chunk, Item, Principal, Visibility


def can_view(
    principal: Principal,
    *,
    visibility: Visibility,
    project_id: str | None,
    author_id: str | None,
    allowed_user_ids: list[str],
    allowed_group_ids: list[str],
) -> bool:
    if principal.is_admin:
        return True
    if author_id is not None and author_id == principal.user_id:
        return True
    if principal.user_id in allowed_user_ids:
        return True
    if allowed_group_ids and set(allowed_group_ids).intersection(principal.group_ids):
        return True
    if visibility is Visibility.TENANT:
        return True
    if visibility is Visibility.PROJECT:
        return project_id is None or project_id in principal.project_ids
    return False


def can_view_item(principal: Principal, item: Item) -> bool:
    return can_view(
        principal,
        visibility=item.acl.visibility,
        project_id=item.project_id,
        author_id=item.author_id,
        allowed_user_ids=item.acl.allowed_user_ids,
        allowed_group_ids=item.acl.allowed_group_ids,
    )


def can_view_chunk(principal: Principal, chunk: Chunk) -> bool:
    return can_view(
        principal,
        visibility=chunk.visibility,
        project_id=chunk.project_id,
        author_id=chunk.author_id,
        allowed_user_ids=chunk.allowed_user_ids,
        allowed_group_ids=chunk.allowed_group_ids,
    )
