import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import prettier from 'eslint-config-prettier';
import globals from 'globals';

export default tseslint.config(
  {
    // `ignores` must be the only key here, otherwise it stops applying globally
    ignores: ['**/dist/**', '**/node_modules/**', '**/.astro/**'],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    languageOptions: {
      globals: { ...globals.node, ...globals.browser },
    },
    rules: {
      '@typescript-eslint/no-unused-vars': [
        'warn',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
      ],
      // only flag a destructuring pattern when every binding in it could be const
      'prefer-const': ['error', { destructuring: 'all' }],
    },
  },
  prettier, // must be last — disables formatting rules
);
