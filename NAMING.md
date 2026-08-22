# Naming

`nano-lm` is the **working name**. The final brand has not been decided.

npm availability, checked 2026-08-22 with `npm view <name>` (404 = unclaimed):

| Candidate      | npm        | Notes                                                                 |
| -------------- | ---------- | --------------------------------------------------------------------- |
| `nano-lm`      | free (404) | Matches the essay's internal `src/nano/` name and the `NanoGPT` class. Generic; "nano" is crowded on npm (`nanoid`, `nano`, …). |
| `glassbox-lm`  | free (404) | Says what the library is *for* — every intermediate is inspectable. Less obvious that it is a runnable model. |
| `tinylm-ts`    | free (404) | Nods at TinyStories and the language. The `-ts` suffix reads as a port of something that exists elsewhere; it does not. |

Unclaimed today does not mean unclaimed at publish time: re-check
immediately before `npm publish`. Also search GitHub and PyPI for the chosen
name to avoid colliding with an unrelated project (e.g. "nanoGPT" is Karpathy's
training repo — a different thing, and a reason to be careful with `nano`).

Whatever the brand, the exported class/type names (`NanoGPT`, `NanoMeta`,
`NanoTensor`) are part of the API surface consumed by Inside the Machine;
renaming them is a breaking change to coordinate with the essay.
