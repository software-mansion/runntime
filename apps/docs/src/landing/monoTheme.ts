/** A grayscale Shiki theme for the landing page code samples: the page has no hue. */

export const monoTheme = {
  name: 'runntime-mono',
  type: 'dark',
  colors: { 'editor.background': '#0e0e0e', 'editor.foreground': '#c6c6c6' },
  tokenColors: [
    {
      scope: ['comment', 'punctuation.definition.comment'],
      settings: { foreground: '#7c7c7c', fontStyle: 'italic' },
    },
    {
      scope: ['keyword', 'storage', 'storage.type', 'storage.modifier', 'variable.language'],
      settings: { foreground: '#ffffff' },
    },
    {
      scope: [
        'entity.name.type',
        'entity.name.class',
        'support.class',
        'entity.other.inherited-class',
      ],
      settings: { foreground: '#ededed' },
    },
    { scope: ['entity.name.function', 'support.function'], settings: { foreground: '#ffffff' } },
    {
      scope: ['string', 'constant.numeric', 'constant.language'],
      settings: { foreground: '#a0a0a0' },
    },
    { scope: ['punctuation', 'meta.brace'], settings: { foreground: '#808080' } },
  ],
};
