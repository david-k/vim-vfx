if exists('g:loaded_vfx')
  finish
endif
let g:loaded_vfx = 1


" Main functions
"-------------------------------------------------------------------------------
function! Vfx() abort
py3 << EOF
import vfx
vfx.init()
EOF

	nnoremap <buffer> <CR> :py3 vfx.open_entry()<CR>
	nnoremap <buffer> <Backspace> :py3 vfx.move_up()<CR>
	nnoremap <buffer> g. :py3 vfx.toggle_dotfiles()<CR>
	nnoremap <buffer> gl :py3 vfx.toggle_details()<CR>
	nnoremap <buffer> <Tab> :py3 vfx.toggle_expand()<CR>
	nnoremap <buffer> - :py3 vfx.quit()<CR>


	autocmd BufUnload <buffer> py3 vfx.on_buf_unload()
	autocmd BufWriteCmd <buffer> py3 vfx.on_buf_save()

	setl conceallevel=2
	setl concealcursor=nvic
	setl textwidth=0
	setl wrapmargin=0
	setl indentexpr=py3eval('vfx.get_indent_for_vim('.v:lnum.')')
endfunction


" Syntax highlighting
"-------------------------------------------------------------------------------
" \zs and \ze mark the start/end of a match, but it seems this does not always
" work as expected in combination with `contained`. Also see
" https://vi.stackexchange.com/questions/14047/how-does-zs-interact-with-syntax-matches

autocmd FileType vfx syntax match VfxLeadingSpace "^[ \t]*" nextgroup=VfxID,VfxAdded
autocmd FileType vfx syntax region VfxDetails start="^\[" end="\][ \t]*" nextgroup=VfxID

autocmd FileType vfx syntax match VfxID "#[0-9]\+:[0-9]\+" contained conceal cchar=#
autocmd FileType vfx syntax match VfxID "#[0-9]\+:[0-9]\+x" contained conceal cchar=# nextgroup=VfxExecutable
autocmd FileType vfx syntax match VfxID "+[0-9]\+:[0-9]\+" contained conceal cchar=+ nextgroup=VfxDirectory
autocmd FileType vfx syntax match VfxID "-[0-9]\+:[0-9]\+" contained conceal cchar=- nextgroup=VfxDirectory

" Matches directory entries (those that end with '/').
" In the search pattern below, \{-1,} denotes non-greedy repetition (with at
" least one element).
" `contained` ensures that this rule will not be considered at the top-level
" but only when explcitly mentioned int a `nextgroup` or `contains`
autocmd FileType vfx syntax match VfxDirectory ".\{-1,}/" contained

autocmd FileType vfx syntax match VfxExecutable ".\+" contained contains=VfxLink,VfxBrokenLink,VfxUnknownLink

autocmd FileType vfx syntax match VfxLink " -> .\+"
autocmd FileType vfx syntax match VfxBrokenLink " ->! .\+"
autocmd FileType vfx syntax match VfxUnknownLink " ->?"

" Matches new entries without a leading ID
autocmd FileType vfx syntax match VfxAdded "[^\[#+\- \t].*" contained


autocmd FileType vfx highlight def link VfxDirectory Directory
autocmd FileType vfx highlight def link VfxExecutable Type
autocmd FileType vfx highlight def link VfxDetails Comment
autocmd FileType vfx highlight def link VfxID SpecialKey
autocmd FileType vfx highlight def link VfxAdded Added
autocmd FileType vfx highlight def link VfxLink Comment
autocmd FileType vfx highlight def link VfxBrokenLink Question
autocmd FileType vfx highlight def link VfxUnknownLink Question
