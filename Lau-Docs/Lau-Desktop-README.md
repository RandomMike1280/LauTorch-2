# Lau Desktop

Lau Desktop is a Python 3.11+ tree-walking interpreter for the documented
core of Lau 5.4.1. It runs normal `.lau` entry scripts and `.laum` modules
without Roblox or a Lua/Luau installation.

The implementation is based on supplied black-box Lau notes rather than a
complete language specification. The precise compatibility choices are
recorded in `COMPATIBILITY.md`, included with the source distribution.

## Run

The package has no runtime dependencies. From this directory:

```powershell
python -m pip install .
lau run path\to\script.lau
```

For editable development, use `python -m pip install -e ".[dev]"`.

Without installing it, set `PYTHONPATH` to `src`:

```powershell
$env:PYTHONPATH = "src"
python -m lau run path\to\script.lau
```

On POSIX shells, the equivalent is:

```sh
PYTHONPATH=src python -m lau run path/to/script.lau
```

Other commands:

```text
lau check script.lau
lau repl
lau script.lau
```

`lau script.lau` is shorthand for `lau run script.lau`. Successful execution
returns exit code 0, Lau runtime failures return 1, and syntax/CLI errors
return 2.

Numeric `for` loops support an optional step, but their body must start on a
new line:

```lau
for i = 1, 10, 2 do
    print(i)
end
```

See `examples/numeric_for_demo.lau` for a runnable version. The compact form
`for i = 1, 10 do print(i) end` is intentionally rejected.

## Example

```lau
func sum(values)
    varol total = 0
    for _, value inpairs(values) do
        total += value
    end
    return total
end

varol values = {1, 2, 3, 4}
print("total=" + sum(values))
```

## Timing

The runtime preserves Lau's measured billing tiers through an injectable
scheduler:

- executed statement: 0.01 seconds by default;
- user function entry: one current statement-cost unit;
- `print`: the documented print-overhead ratio above its statement;
- `task.wait(N)`: the documented wait-overhead ratio plus `N`.

The reported game baseline was approximately 0.065 seconds per statement.
Desktop Lau defaults to 0.01, and a script can change the cost for subsequent
statements with a source directive:

```lau
@cost(t=0.0025)
```

The default scheduler enforces charges against a cumulative wall-clock
deadline. This makes `--statement-cost` and `@cost(t=n)` directly observable
without multiplying the operating system's minimum sleep duration across
every statement. Sub-millisecond deadlines use a short precision spin; larger
ones sleep normally. Large configured values can therefore create long waits.
Use `--virtual-time` for fast simulation: charges still advance
`task.clock()`, but do not sleep or spin.

```text
lau run script.lau --statement-cost 0.01
lau run script.lau --cost-scale 0
lau run script.lau --wait-scale 0
lau run script.lau --realtime
lau run script.lau --virtual-time
lau run script.lau --max-statements 100000
lau run script.lau --module-path shared-modules
lau run script.lau --seed 42
```

`--cost-scale` multiplies both the default statement cost and later `@cost`
values. `--realtime` explicitly selects the default wall-clock behavior,
while `--virtual-time` disables sleeping. `--wait-scale` independently
controls the explicit `N` portion of `task.wait(N)`.

Costs below the interpreter's own execution time cannot make a statement run
faster than that execution time. They therefore converge on the same measured
`task.clock()` baseline as zero cost instead of being rounded up to the OS
sleep floor. `ExecutionResult.charged_seconds` retains the exact nominal
charge independently of that physical limit.

## Python API

```python
from lau import Interpreter, RuntimeConfig

runtime = Interpreter(
    RuntimeConfig(statement_cost=0.01, seed=42, max_statements=100_000)
)
result = runtime.run_source('print(math.random(1, 10))', "example.lau")

print(result.stdout, end="")
if not result.success:
    print(result.stderr, end="")
```

The public API also exports `ExecutionResult`, `run_source`, `run_file`, and
`check_source`.

## Tests

```powershell
python -m pytest
```

The suite reconstructs the core probes and examples from both source
documents, including numeric behavior, parser exclusions, closures,
multi-return values, lists, imports, patterns, diagnostics, CLI behavior, and
the cost profile and both wall-clock and virtual timing modes.

## Build Distributions

The project uses `setuptools.build_meta` with a `src/` package layout. Build
and validate both distribution formats with:

```powershell
python -m build
python -m twine check --strict dist\*
```

When building offline with the declared `setuptools>=77` requirement already
installed, use `python -m build --no-isolation`.

This creates a pure-Python wheel and source archive under `dist/`. The wheel
contains only the `lau` runtime package and console entry point. The source
archive also includes tests, examples, compatibility notes, the changelog,
and the MIT license.

## License

Lau Desktop is distributed under the MIT License. See `LICENSE` in the source
distribution.
