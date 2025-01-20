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
	nnoremap <buffer> _ :py3 vfx.open_vim_cwd()<CR>
	nnoremap <buffer> ´ :py3 vfx.change_vim_cwd()<CR>
	nnoremap <buffer> g. :py3 vfx.toggle_dotfiles()<CR>
	nnoremap <buffer> gl :py3 vfx.toggle_details()<CR>
	nnoremap <buffer> <Tab> :py3 vfx.toggle_expand()<CR>
	nnoremap <buffer> - :py3 vfx.quit()<CR>


	autocmd BufUnload <buffer> py3 vfx.on_buf_unload()
	autocmd BufWriteCmd <buffer> py3 vfx.on_buf_save()
	autocmd InsertLeave <buffer> py3 vfx.display_changes()
	autocmd TextChanged <buffer> py3 vfx.display_changes()

	" TODO Define `autocmd FileExplorer`. This makes it possible to do `:edit <DIRECTORY>` like netrw

	setl conceallevel=2
	setl concealcursor=nvic
	setl textwidth=0
	setl wrapmargin=0
	setl indentexpr=py3eval('vfx.get_indent_for_vim('.v:lnum.')')
	setl cursorline
endfunction
