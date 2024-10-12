" \zs and \ze mark the start/end of a match, but it seems this does not always
" work as expected in combination with `contained`. Also see
" https://vi.stackexchange.com/questions/14047/how-does-zs-interact-with-syntax-matches

syntax match VfxLeadingSpace "^[ \t]*" nextgroup=VfxID,VfxAdded
syntax region VfxDetails start="^\[" end="\][ \t]*" nextgroup=VfxID

syntax match VfxID "|[0-9]\+:[0-9]\+_" contained conceal cchar=|
syntax match VfxID "|[0-9]\+:[0-9]\+x" contained conceal cchar=| nextgroup=VfxExecutable
syntax match VfxID "+[0-9]\+:[0-9]\+_" contained conceal cchar=+ nextgroup=VfxDirectory
syntax match VfxID "-[0-9]\+:[0-9]\+_" contained conceal cchar=- nextgroup=VfxDirectory

" Matches directory entries (those that end with '/').
" In the search pattern below, \{-1,} denotes non-greedy repetition (with at
" least one element).
" `contained` ensures that this rule will not be considered at the top-level
" but only when explcitly mentioned int a `nextgroup` or `contains`
syntax match VfxDirectory ".\{-1,}/" contained

syntax match VfxExecutable ".\+" contained contains=VfxLink,VfxBrokenLink,VfxUnknownLink

syntax match VfxLink " -> .\+"
syntax match VfxBrokenLink " ->! .\+"
syntax match VfxUnknownLink " ->?"

" Matches new entries without a leading ID
syntax match VfxAdded "[^\[|+\- \t].*" contained


highlight def link VfxDirectory Directory
highlight def link VfxExecutable Type
highlight def link VfxDetails Comment
highlight def link VfxID SpecialKey
highlight def link VfxAdded Added
highlight def link VfxLink Comment
highlight def link VfxBrokenLink Question
highlight def link VfxUnknownLink Question
