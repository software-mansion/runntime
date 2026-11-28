# Drafts

Content that is written but not published yet. Nothing in this directory is
routed — it sits outside `src/content/docs/`, which is the base directory
Starlight's content collection loads from.

`core/` holds the `runntime/core` reference. It is deliberately unpublished for the
initial release. To publish it again: move the directory back to
`src/content/docs/core/` and restore the `Core` topic in
`starlightSidebarTopics(...)` in `astro.config.mjs`.
