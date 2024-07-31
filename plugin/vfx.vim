if exists('g:loaded_vfx')
  finish
endif
let g:loaded_vfx = 1

let s:expanded_dirs = {}
let s:cursor_line = 1


" Main functions
"-------------------------------------------------------------------------------
function! Vfx() abort
	" Find free buffer name
	let l:buf_name = "[Directory] ". getcwd()
	let l:counter = 0
	while 1
		let l:test_name = s:MakeName(l:buf_name, l:counter)
		if !bufexists(l:test_name) || !getbufinfo(l:test_name)[0]["listed"]
			break
		endif

		let l:counter += 1
	endwhile
	let l:buf_name = s:MakeName(l:buf_name, l:counter)

	" The directory buffer should have no influence on the alternative file.
	" To this end, we need to update the alt file in two situations:
	" 1. When closing the directory buffer:
	"    - In this case we restore the state as if Vfx() was never called,
	"      i.e, we need to restore the alt file to the one before calling
	"      Vfx(). We do this with a autocmd on BufUnload.
	" 2. When opening a file:
	"    - In this case we want to set the alt file to the file that was
	"      active when calling Vfx(). We can achieve that by using :keepalt.
	"      (See VfxOpen() below. Note that we need to disable the BufUnload
	"      autocmd so that it doesn't mess with the alt file)

	" We need to query the alt file *before* switching the buffer
	let l:alt_buf = bufnr("#")

	" Create new buffer and set options
	let l:buf_no = bufadd(l:buf_name)
	execute "silent buffer ". l:buf_no
	setlocal bufhidden=delete
	setlocal buftype=nofile
	setlocal noswapfile
	setlocal buflisted
	setlocal ft=fileexplorer

	let b:current_dir = getcwd()
	let b:expanded_dirs = copy(s:expanded_dirs)
	let b:show_dotfiles = 0
	let b:alternate_buf = l:alt_buf

	nnoremap <buffer> <CR> :call VfxOpen()<CR>
	nnoremap <buffer> <Backspace> :call VfxDirUp()<CR>
	nnoremap <buffer> g. :call VfxToggleDotfiles()<CR>
	nnoremap <buffer> <Tab> :call VfxToggleExpand()<CR>

	autocmd BufUnload <buffer> call s:OnBufUnload()

	call VfxUpdate()
	execute "normal ". s:cursor_line ."G"
endfunction

function! VfxOpen() abort
	let l:filepath = b:current_dir ."/". s:GetPathAt(line("."))
	if isdirectory(l:filepath)
		let b:current_dir = l:filepath
		call VfxUpdate()
	elseif filereadable(l:filepath)
		autocmd! BufUnload <buffer>
		call s:OnExit()
		execute "keepalt edit ". l:filepath
	endif
endfunction

function! VfxDirUp() abort
	" If the path ends with a path separator like /some/path/ then :h only
	" removes the path separator
	let b:current_dir = fnamemodify(trim(b:current_dir, "/", 2), ":h") 
	call VfxUpdate()
endfunction

function! VfxToggleExpand() abort
	let l:line_no = line(".")
	let l:filepath = trim(b:current_dir ."/". s:GetPathAt(l:line_no), "/", 2)
	if isdirectory(l:filepath)
		let b:expanded_dirs[l:filepath] = !s:IsExpanded(l:line_no)
	endif
	call VfxUpdate()
endfunction

function! VfxToggleDotfiles() abort
	let b:show_dotfiles = !b:show_dotfiles
	call VfxUpdate()
endfunction

function VfxUpdate() abort
	let l:items = s:FlattenDirItems(s:GetDirItemsTree(b:current_dir, b:expanded_dirs))
	let l:cur_line = line(".")
	normal ggdG
	call append(0, l:items)
	normal ddgg
	execute "normal ". l:cur_line ."G"
endfunction


autocmd FileType fileexplorer syntax match Directory ".*/$"


" Utils
"-------------------------------------------------------------------------------
function! s:OnBufUnload() abort
	call s:OnExit()
	if bufexists(b:alternate_buf)
		let @# = b:alternate_buf
	endif
endfunction

function s:OnExit() abort
	let s:expanded_dirs = copy(b:expanded_dirs)
	let s:cursor_line = line(".")
endfunction

function! s:GetDirItemsTree(dir, expanded_dirs) abort
	let l:entries = readdirex(a:dir)

	let l:files = []
	let l:folders = []
	for l:entry in l:entries
		if !b:show_dotfiles && l:entry["name"][0] == "."
			continue
		endif

		let l:type = l:entry["type"]
		if l:type == "file" || l:type == "link"
			call add(l:files, {"name": l:entry["name"], "type": l:type})
		elseif l:type == "dir" || l:type == "dlink"
			let l:subpath = s:JoinPaths(a:dir, l:entry["name"])
			let l:entry = {"name": l:entry["name"], "type": l:type, "children": []}
			if get(a:expanded_dirs, l:subpath, 0)
				let l:entry["children"] = s:GetDirItemsTree(l:subpath, a:expanded_dirs)
			endif
			call add(l:folders, l:entry)
		endif
	endfor

	return extend(l:folders, l:files)
endfunction

function! s:FlattenDirItems(dir_tree, indent = "") abort
	let l:result = []
	for l:item in a:dir_tree
		let l:type = l:item["type"]
		if l:type == "file" || l:type == "link"
			call add(l:result, a:indent . l:item["name"])
		elseif l:type == "dir" || l:type == "dlink"
			call add(l:result, a:indent . l:item["name"] ."/")
			call extend(l:result, s:FlattenDirItems(l:item["children"], a:indent . "\t"))
		endif
	endfor

	return l:result
endfunction

function! s:GetPathAt(line_no)
	let l:cur_line = a:line_no
	let l:parts = [trim(getline(l:cur_line), "/ \t")]
	while 1
		let l:cur_line = s:GoToParent(l:cur_line)
		if !l:cur_line
			break
		endif

		call add(l:parts, trim(getline(l:cur_line), "/ \t"))
	endwhile

	return join(reverse(l:parts), "/")
endfunction

function! s:GoToParent(line_no)
	let l:cur_line = a:line_no
	let l:cur_indent = s:GetIndent(getline(l:cur_line))
	while 1
		let l:cur_line -= 1
		if l:cur_line == 0
			break
		endif

		if s:GetIndent(getline(l:cur_line)) < l:cur_indent
			return l:cur_line
		endif
	endwhile

	return 0
endfunction

function! s:IsExpanded(line_no)
	return s:GetIndent(getline(a:line_no + 1)) > s:GetIndent(getline(a:line_no))
endfunction

function! s:GetIndent(str)
	let l:indent = 0
	for l:c in a:str
		if l:c == "\t"
			let l:indent += 1
		else
			break
		endif
	endfor

	return l:indent
endfunction

function! s:JoinPaths(p1, p2) abort
	return trim(a:p1, "/", 2) ."/". a:p2
endfunction

function! s:MakeName(name, counter) abort
	if a:counter == 0
		return a:name
	endif

	return a:name ." [". a:counter ."]"
endfunction
