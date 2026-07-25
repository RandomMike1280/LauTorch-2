# LAU Language Guide for Drone Farming

This guide covers the basics of the `.lau` scripting language used for automating drone farming based on the available examples.

## Language Dialect — Notable Findings

Lau is a heavily customised dialect of Lua. Many standard Lua features are removed, and several runtime behaviours are subtle enough to cause silent bugs. Every finding in this section has been confirmed against the Lau interpreter; the corresponding test artefacts are recorded in `LAU_SPEC.md`. Findings marked *docs correction* are discrepancies between what Lau actually does and what some community documentation states.

### Numeric Model

- **Every number is a 64-bit IEEE 754 float64.** There is no integer type. `type(1)` and `type(1.5)` both return `"Number"`. Values that look like integers and values that result from bitwise arithmetic are both stored as floats.
- **Division always produces a float.** `5 / 2 == 2.5`, not `2`. There is no `//` integer-division operator and no equivalent. Use `math.floor(a / b)` if you need the integer quotient.
- **Modulo follows IEEE 754 truncated-division semantics.** The result carries the sign of the *divisor*, not the dividend: `-7 % 3 == 2`, `7 % -3 == -2`, `-7 % -3 == -1`. This differs from Python's floor-division convention.
- **Division by zero is a fatal runtime error.** The interpreter aborts the program with `Operation Error: It is not divisible by zero.` Likewise, `5 % 0` aborts with `It cannot be zero.` There is no `inf`, no `NaN`, no IEEE 754 infinity-arithmetic escape hatch. Validate inputs before dividing.
- **No hexadecimal literals.** `0xFF` is parsed as a function call. Use `tonumber("0xFF")` instead.
- **No scientific notation.** `1e308` is likewise parsed as a function call. No `E`-notation support at the lexer level.
- **No bitwise operators whatsoever.** No `&`, `|`, `~`, `>>`, `<<`, no `bnot`, no `bor`. Bit manipulation has to be done by hand through floating-point arithmetic, which is only reasonable within the safe-integer range.

### Types

- **`type()` returns six strings, and only six:** `"Number"`, `"String"`, `"Boolean"`, `"null"`, `"List"`, `"Color"`. There is no `"integer"`, `"float"`, `"table"`, `"function"`, `"userdata"`, or `"thread"`. A defensive `type(x) == "integer"` guard will never fire.
- **Note that `"null"` is lowercase and is the literal return value, not a type-name in the standard sense.** Compare `type(x) == "null"`, not `== "Nil"`.
- **Lists are strictly 1-indexed.** Index 0 and negative indices return `null`; out-of-bounds reads return `null` without raising an error. Use `#list` for the length.

### Functions and Errors

- **The keyword is `func`, not `function`; the variable-declaration keyword is `varol`, not `local`.** Both lowercase variants are syntax errors — they are parsed as undefined identifiers.
- **Functions must be called with parentheses.** A bare reference like `drone.move` is not a call. To capture and invoke later, assign to a variable: `varol myMove = drone.move; myMove(Enum.Direction.South)`.
- **Varargs (`...`) are not supported.** Functions cannot accept a variable number of positional arguments. To forward multiple values, pass a list and index into it manually.
- **Closures and multi-return values work as in Lua**, with one exception: `pcall` on success returns `ok` and exactly **one** additional value (the first of the inner function's returns). If the inner function returned four values, `pcall` passes through three and silently drops one. Wrap results in a list at the callee site if you need them all.
- **No `xpcall`, no `assert`, no `select`.** Implement equivalent helpers yourself if you need them.

### Iteration and Operators

- **Only `inpairs` exists.** There is no `pairs`, no `next`, no `unpack`, no `table.unpack`, no `list.unpack`. To iterate over a map-like table with string keys, build a parallel list of keys and iterate over that list with `inpairs`.
- **The colon-method sugar is broken for built-in modules.** `string.sub(s, ...)` works correctly, but `s:sub(1, 5)` does not — the parser attempts a property read on the string and fails. Always use the explicit module form: `string.sub(s, ...)`, `list.find(arr, v)`, `math.clamp(x, lo, hi)`. The colon syntax **does** remain valid for event listeners (`player.chatted:connect(...)`); the breakage applies only to the strings / lists / math sugar.
- **`string.gsub` does not accept a function as the replacement argument.** Only `(string, pattern, stringReplacement)` is valid. To produce dynamic replacements, pre-compute the replacement string and pass it as the third argument.

### Runtime Limits and Billing

- **The list-size cap is 800, not 8000.** *(docs correction)* Some Lau documentation states a cap of 8000 elements. The actual hard limit is 800, and exceeding it is fatal — writing to `arr[801]` aborts with `Assignment Error: The list is full! Maximum 800 elements.`. Confirm any long-running script stays well below this bound.
- **Billing is per *statement*, not per operation.** Each statement costs approximately 0.06–0.07 s regardless of how many operations it contains. A single complex line is cheaper than the same operations split across multiple lines. For tight loops in farming paths, inline aggressively.
- **Function-call overhead is significant** — roughly 3× the cost of an inlined statement (≈0.20 s per call vs ≈0.07 s per statement). In the inner loops of common farming scripts, inlining the function body onto the call site can save real wall-clock time.
- **`print(...)` is its own cost tier, between simple statements and function calls.** Averaged over 5×100 iterations, a statement containing `print(...)` costs ≈0.113 s/stmt (tight 0.112–0.114 s band); the print contribution above the simple-statement baseline is ≈0.05 s; string-concat arguments to `print` add essentially nothing (≈0.002 s). One-shot measurements may report higher values than the average; if you're optimising a print-heavy inner loop, batch the output — write to a list or string buffer and `print` once per N iterations — rather than calling `print` every iteration. This matters because `print` is roughly 1.7× as expensive as a simple statement and only ≈0.6× as expensive as a function call, so it's easy to underestimate its cost in logging-heavy code.
- **`task.wait(N)` adds approximately `N` seconds of wall-clock time on top of a per-call fixed overhead.** A 100-iteration benchmark shows `task.wait(0)` ≈ 0.13 s/call and `task.wait(0.05)` ≈ 0.08 s/call, with the per-call difference matching the argument; there is no observable minimum-quantum floor. The `N`-component is imprecise at sub-100 ms scale and warrants averaging over many calls. `task.wait` returns `null` on every call and is not a stopwatch — use `task.clock()` for timing.

### Caveats Specific to This Project's APIs

- **The drone-related object surface is largely unique to this game** (drone, droneV2, market, garden, cache, display, http) and is documented in the project's own API reference rather than in the Lau language specification. Findings in `LAU_SPEC.md` cover language-level behaviour; project-API behaviour is recorded alongside the relevant sections later in this guide.

## Basic Syntax

### Variables
Declare variables using the `varol` keyword:
```lau
varol item = player.getItem(1)
varol x, z = drone.getPosition() -- Multiple assignment is supported
varol count, maxCount = 0, 20
```

Use `varol` only when declaring new variables. When updating an existing variable, assign to it directly:
```lau
varol seedBudget = 10
seedBudget = seedBudget - 1
```

### Comments
Use `--` for single-line comments. They must be on a separate line or at the end of a statement.
```lau
-- This is a comment
varol result = math.abs(-50) -- Prints 50
```

### Strings
Strings are joined (concatenated) using the `+` operator, not the `..` operator typically found in Lua.
```lau
print("Date: " + task.date())
print("Budget: " + seedBudget)
```

Numbers can usually be concatenated with strings directly using `+` in this environment.

### Lists and Dictionaries
`.lau` uses a unified structure for both ordered lists (arrays) and key-value pairs (dictionaries). Note that lists are **1-indexed** (the first element is at index 1).

**Defining Lists:**
```lau
-- Dictionary style
varol droneData = {
    ["Speed"] = 15,
    ["Mode"] = "Automatic"
}

-- Array style (Ordered List)
varol fruits = {"Apple", "Pear", "Banana"}
```

**Accessing and Modifying:**
You do not use `varol` when updating an existing list.
```lau
varol inventory = {"Wheat", "Corn", "Tomato"}
print(inventory[2]) -- Retrieves "Corn"
print(#inventory) -- The '#' operator gets the length of the list

inventory.new = "Potato" -- Add/update using dot notation
inventory[1 + 2] = "Watermelon" -- Add/update dynamically using index (updates 3rd item)

-- Deleting elements uses 'null' instead of 'nil'
droneData.Speed = null
```

Table indexes must be numbers or strings. Avoid using an object/table as a key:
```lau
varol fieldState = {}
fieldState["-1,2"] = "planted" -- Valid
fieldState[3] = "ready" -- Valid

-- Invalid if coord is a table/object:
-- fieldState[coord] = "planted"
```

You can use string keys with either dot notation or bracket notation. Use bracket notation when the key contains special characters:
```lau
stats.Restocks += 1
stats["Watering_Can"] += 1
stats["MaxStock"]["SprinklerV3"] = 5
```

### Positional Tables and Placeholders
For module exports, this project often uses ordered tables and numeric indexes instead of named fields. This avoids parser/runtime problems seen with some named return tables and keeps module contracts stable:
```lau
return {
    init,        -- [1]
    hasSeed,     -- [2]
    buyMax,      -- [3]
    null,        -- [4] reserved/removed slot
    doTile,      -- [5]
    sweep        -- [6]
}
```

When removing a function from an exported module table, keep a `null` placeholder if callers might rely on later numeric slots. This prevents shifting every function index.

Inside normal dictionaries, named string keys are still useful for structured data:
```lau
varol gearStats = {
    ["Restocks"] = 0,
    ["MaxStock"] = {
        ["SprinklerV3"] = 0
    }
}
```

### Control Flow
The language primarily uses Lua-like block structures (`if / then / end`, `while true do`). While C-like syntax elements (`if (condition) { ... }`) are supported, they are known to be poorly implemented and buggy. **Always prefer the Lua style for stability.**

**Logical and Relational Operators:**
*   Use uppercase `AND`, `OR`, and `NOT` for logical conditions.
*   Use `~=` for "not equal" (e.g., `if count ~= 5 then`).
*   Use `==` for equality and normal comparisons like `<`, `<=`, `>`, and `>=`.
*   `true`, `false`, and `null` are the common boolean/empty values.
*   `+=` and `-=` are supported for incrementing/decrementing variables.
*   `break` exits the current loop.

```lau
if item AND it.Type == "Seed" then
    -- code
elseif NOT item then
    -- code
end

while true do
    -- code
    if count >= 10 then break end
end

count += 1
count -= 1

-- Numeric For Loop
for i = 1, 5 do 
    print("Number:", i) 
end 

-- For Loop over Lists (use 'inpairs')
varol fruits = {"Apple", "Pear", "Cherry"} 
for index, fruit inpairs(fruits) do 
    print(index + ". fruit: " + fruit) 
end
```

Short one-line blocks are supported and used in this project:
```lau
if buyingSeeds then return end
if market.buyGear(Enum.Gear.Watering_Can) == false then break end
```

### Defining Functions
You can define standard functions or assign anonymous functions to variables using the `func` keyword and closing with `end`.

```lau
-- Standard definition
func add(a, b) 
    return a + b 
end 

-- Anonymous function assigned to a variable
varol multiply = func(x, y) 
    return x * y 
end 
```

### Functions and Calling
To call a function, you must use parentheses. If you omit parentheses, you are referencing the function, not calling it.
```lau
drone.move(Enum.Direction.East) -- CORRECT: Calls the function
varol myMove = drone.move -- Assigns the function reference to a variable
myMove(Enum.Direction.South) -- Calls the referenced function
```

### Modules and Imports
You can split your code into separate `.laum` module scripts and import them into your main script using the `req()` function.
```lau
-- Import a module
varol farmingModule = req("FarmingOperations.laum")
```

Modules return values with `return`. A common pattern is to return a table of functions:
```lau
-- Movement.laum
func init(config)
    -- setup
end

func moveTo(x, z)
    droneV2.goto(x, z)
end

return {
    init = init,
    moveTo = moveTo,
}
```

The caller may be able to access returned functions by name:
```lau
varol movement = req("Movement.laum")
movement.init(config)
movement.moveTo(0, 0)
```

This project uses numeric access (`movement[1]`, `farming[6]`) because returned module tables are known to work that way in this runtime. Named access is easier to read, but verify it in-game before converting existing modules.

Only main `.lua` / `.lau` scripts should use pragmas like `--!ndrone`. Module files (`.laum`) are loaded with `req()` and should just return their functions/data.

### Stable Module Slot Contracts
When a project uses numeric module slots, document the slot contract and update it carefully. For example, the current farming module exports service-related functions through fixed indexes:
```lau
farming[6](seedBudget, true)  -- sweep
farming[9]()                  -- garden scan
farming[12](null, seedBudget) -- plant pass
farming[16](commands[2])      -- service hook setter
```

This style is compact, but it makes accidental slot shifts dangerous. Prefer adding `null` placeholders over deleting old slots.

## Events

Events are listeners that run a function when something happens. They use callback functions inside parentheses:
```lau
player.chatted:connect(func(message)
    print("Player said: " + message)
end)
```

An event has three parts:
*   The event to listen to, such as `player.chatted`.
*   The listener type after the colon, such as `:connect` or `:once`.
*   The callback function, such as `func(message) ... end`.

Event callbacks run immediately when the event is triggered. In observed behavior, callbacks from events such as `market.changedSeedStock` and `player.chatted` do not wait for the main program loop to reach them. They run in a separate pseudo-thread, which means events can be used for parallel-style reactive processing: market buying, hotkeys, command flags, and other reactive work can happen while the main farming loop continues.

Because event callbacks can run concurrently with the main loop, protect shared state with simple guard flags when the callback performs work that can overlap.

### `:connect`
`:connect` permanently listens and runs the callback every time the event fires.
```lau
player.chatted:connect(func(message)
    if message == "status" then
        player.alert("Running")
    end
end)
```

Do not create permanent listeners inside a `while true do` loop. That creates a new active listener each loop iteration and can overflow the number of active events.

### `:once`
`:once` listens a single time, runs the callback once, then stops listening.
```lau
player.chatted:once(func(item)
    if string.find(item, "Apple") then
        market.buySeed(Enum.Seed.Apple)
    end
end)
```

Use `:once` when you only want one response to a prompt or one future event.

### Event Timing
With `--!ndrone`, event callbacks can run while the drone is still moving or working. This is useful for market refresh handlers:
```lau
market.changedSeedStock:connect(func()
    player.alert("Market seed stocks refreshed!")
    market.buySeed(Enum.Seed.Lotus)
end)

market.changedGearStock:connect(func()
    player.alert("New gears arrived!")
    market.buyGear(Enum.Gear.Watering_Can)
end)
```

Use guard flags if an event handler might take time and you want to prevent overlapping runs:
```lau
varol buyingSeeds = false

market.changedSeedStock:connect(func()
    if buyingSeeds then return end
    buyingSeeds = true
    market.buySeed(Enum.Seed.Lotus)
    buyingSeeds = false
end)
```

This separate-thread behavior is powerful, but it also means physical drone actions can conflict if multiple callbacks and the main loop command the drone at the same time. Prefer using events to set flags or trigger non-movement work unless the callback owns the full action sequence.

### Event Concurrency Limits
Lau does not have a mutex lock primitive. There is no built-in `lock`, `mutex`, or atomic section for protecting shared state. Use simple boolean flags when you need to prevent overlapping work.

Each event listener appears to allow only one active callback instance at a time. For example:
```lau
some_event:connect(func(n)
    -- do something slow
end)
```

If `some_event` triggers once and the callback is still running, triggering that same event again will not start a second copy of the same callback until the first callback finishes. In other words, each event can spawn only one pseudo-thread at a time.

This means event callbacks are parallel relative to the main loop, but not infinitely re-entrant per event listener. If you need repeatable work, keep callbacks short, set module-level flags, and let the main loop or a service step do the long-running processing.

### Player Input Event
`player.input:connect` listens to keyboard input and passes the pressed key as an `Enum.KeyCode` value:
```lau
player.input:connect(func(k)
    print(k)
end)
```

Example output while pressing keys:
```text
Enum.KeyCode.K
Enum.KeyCode.S
Enum.KeyCode.W
Enum.KeyCode.A
Enum.KeyCode.F
Enum.KeyCode.D
```

This can be used for hotkeys instead of relying only on chat commands:
```lau
varol running = false

player.input:connect(func(k)
    if k == Enum.KeyCode.K then
        running = NOT running
        if running then
            player.alert("Running")
        else
            player.alert("Paused")
        end
    end
end)
```

Like chat and market events, input callbacks fire as soon as the key is pressed. Keep hotkey callbacks short: toggle flags, update modes, or request a scan. Avoid long movement loops directly inside `player.input:connect`, because repeated key presses can create overlapping work.

Side note: this event was discovered from the game's loading screen hints, and it is useful for making scripts feel more interactive than chat-only command systems.

### Chat Commands Should Not Own Long-Running Loops
Chat event callbacks can start or stop service flags, but avoid putting a permanent `while true do` loop directly inside `player.chatted:connect`. A long-running callback can block later command processing or compete with the main loop.

Prefer this pattern:
```lau
varol hRun = false

p.chatted:connect(func(msg)
    if msg == "init" then
        hRun = true
        return p.alert("Init on")
    end
    if msg == "init off" then
        hRun = false
        return p.alert("Init off")
    end
end)

func step()
    if hRun then
        harvStep()
    end
end
```

Then call `step()` from the main loop and from long waits. This gives background behavior without trapping execution inside the chat callback.

## Pragmas and Asynchronous Execution

Pragmas are special instructional commands that tell the `.lau` engine how to process your code. They must be placed at the **very top** of your main `.lau` script (they do not work inside `.laum` module scripts).

### The `--!ndrone` Pragma
By default, `.lau` operates synchronously. When you issue a drone command (like `drone.doFlip()`), the script pauses until the animation finishes. 

Adding `--!ndrone` at the top of your script enables **Asynchronous (Non-Blocking) mode**. The engine will issue the command and instantly skip to the next line.

```lau
--!ndrone
drone.doFlip()
print("hi") -- Prints instantly while the drone is still flipping!
```

### The Overlapping Problem
In async mode, if you issue a command while the drone is busy, **the new command is completely ignored.**

```lau
--!ndrone
drone.doFlip() -- Starts flipping
drone.doFlip() -- IGNORED! Drone is already busy.
```

To safely use async mode, you must manually check the drone's status using `drone.status()`:
```lau
--!ndrone

while true do
    -- Only send commands if the drone is resting
    if drone.status() == Enum.DroneStatus.Sleep then
        drone.doFlip()
    end
    
    -- You can run other background calculations here!
    
    task.wait(0.1) -- Always include a small wait to prevent crashes in tight loops
end
```

In async scripts, wrap physical drone commands with a status wait if the next line depends on that action finishing:
```lau
func waitDrone()
    while drone.status() ~= Enum.DroneStatus.Sleep do
        task.wait(0.05)
    end
end

waitDrone()
droneV2.goto(2, -1)
waitDrone()
drone.harvest()
waitDrone()
```

### Cooperative Service Hooks
If the main loop can spend time inside farming functions, pass a service callback into those functions and call it during waits or long iterations:
```lau
varol sv = null

func setService(f)
    sv = f
end

func waitDrone()
    while drone.status() ~= Enum.DroneStatus.Sleep do
        if sv then
            sv()
        end
        task.wait(0.03)
    end
end
```

This project uses that pattern so autosell and fruit harvesting continue while crop waits, movement waits, or plant passes are running.

## Core Objects and APIs

### The `cache` Object
`cache` stores small persistent string-like values for the current player. This project uses it for the main state machine and command handoff:
```lau
cache.set("S", "F")
varol st = cache.get("S")
if st == null then st = "F" end
```

Known current keys in this project:
*   `S`: farming state (`F`, `A`, or `P`).
*   `SS`: seed-stock event status.
*   `GS`: gear-stock event status.
*   `C`: command flag, such as `SCAN`.
*   `M`: current mode marker.

Important runtime constraint: the cache has a small per-player key limit. In this project, adding a sixth key caused `Cache limit exceeded! Max 5 keys allowed per player.` Keep transient flags in module-local variables instead of adding more cache keys.

### The `drone` Object
Controls the automation drone's actions and retrieves data about its current tile.

*   **Farming Actions**
    *   `drone.plant(Enum.Seed.[Type])`: Plants a specific seed on the current tile.
    *   `drone.canCrop()`: Returns a boolean indicating if the plant on the current tile can be cropped (cut from root).
    *   `drone.crop()`: Collects crops like Pumpkin, Wheat, Potato, etc.
    *   `drone.canHarvest()`: Returns a boolean indicating if a fruit-bearing tree can be harvested.
    *   `drone.harvest()`: Collects fruit from fruit-bearing trees.
*   **Plant Data**
    *   `drone.getPlant()`: Returns a plant object for the current tile, or `null` if no plant is present. Known properties include `HasFruit`.
    *   `drone.getPlantHasFruit()`: Returns a boolean indicating if the plant has fruit.
    *   `drone.getPlantPercent()`: Returns the growth percentage of the plant itself.
    *   `drone.getFruitPercent()`: Returns the growth percentage of the fruit.
*   **Movement & Position**
    *   `drone.move(Enum.Direction.[Direction])`: Moves the drone one unit in the specified direction (only North, South, East, West).
    *   `drone.doFlip()`: Makes the drone perform a backflip.
    *   `drone.getPosition()`: Returns both X and Z coordinates (`varol x, z = drone.getPosition()`).
    *   `drone.getPositionX()`: Returns only the X coordinate.
    *   `drone.getPositionZ()`: Returns only the Z coordinate, if available in the current runtime.
*   **State & Status**
    *   `drone.status()`: Returns the drone's current state (e.g., `Enum.DroneStatus.Busy` or `Enum.DroneStatus.Sleep`).
    *   `drone.useItem(Enum.Gear.[GearType])`: Commands the drone to use an item, such as a Watering Can.

### The `droneV2` Object
The V2 drone has advanced movement and tile inspection capabilities that read machine and soil buff data.
*   **Advanced Movement**
    *   `droneV2.goto(x, z)`: Commands the drone to travel directly to the specified X and Z coordinates.
    *   `droneV2.swap(Enum.Direction)`: Swaps positions with the plant (or empty space) on the adjacent tile.
*   **Tile Inspection & Gear Data**
    *   `droneV2.isLocked()`: Returns a boolean indicating if the plant on the tile is locked (cannot be swapped).
    *   `droneV2.hasGear()`: Returns a boolean indicating if a machine is currently placed on the tile.
    *   `droneV2.getGear()`: Returns a comprehensive object containing all machine details and soil buff data.
    *   `droneV2.getGearName()`: Returns the specific name of the gear.
    *   `droneV2.getGearDuration()`: Returns its remaining active duration in seconds.
*   **Soil Buffs**
    *   `droneV2.getFertilizer()` / `droneV2.getManualWater()` / `droneV2.getMachineWater()`: Returns an object with `Duration` (remaining seconds) and `Multi` (effectiveness multiplier).
    *   `droneV2.getLightning()`: Returns the remaining duration of the lightning rod protection effect in seconds.

Example water check:
```lau
varol water = droneV2.getManualWater()
if water AND water.Duration == 0 then
    drone.useItem(Enum.Gear.Watering_Can)
end
```

### The `player` Object
Accesses player inventory, stats, and UI interactions.

*   **Inventory & Wealth**
    *   `player.getItem(slotNumber)`: Returns the item in the specified inventory slot (e.g., slot 1 is the first hotbar slot). The item object has properties like `Type`, `Name`, `Amount`, and sometimes `OriginalKey`.
    *   `player.getInventory()`: Returns the entire inventory as a list (table).
    *   `player.getInventorySize()`: Returns the total number of all (Fruit, Seeds, etc.) item types in the inventory.
    *   `player.getFruitCount()`: Returns the total number of fruit item types in the inventory.
    *   `player.getFruitCapacity()`: Returns the max number of fruit item types in the inventory.
    *   `player.scrap()`: Returns the total amount of scrap (currency) the player owns.
    *   `player.calculateFinalScrap(basePrice)`: Returns the actual scrap earned after multipliers (e.g., events).
    *   `player.getTileNumber()`: Returns the player's land size (upgrade level).
*   **UI & Events**
    *   `player.alert("Message")`: Displays an alert message to the player.
    *   `player.chatted:connect(func(message) ... end)`: Event listener triggered when the player types a chat command.
    *   `player.chatted:once(func(message) ... end)`: Event listener triggered by the next chat message only.
*   **Location and Positioning**
    *   `player.getCurrentTile()`: Returns the grid coordinates (X, Z) of the tile the player is currently standing on. If the player is outside the farm area, it returns `null`.
*   **Camera and Player Control**
    *   The following functions act as both Getters and Setters. If you provide an argument, they change the state. If you leave the parentheses empty `()`, they return the current state.
    *   `player.camera(Enum.Camera?)`: Sets or gets the camera's target. Example Set: `player.camera(Enum.Camera.Drone)` Example Get: `varol currentCam = player.camera()`
    *   `player.cameraMode(Enum.CameraMode?)`: Sets or gets the camera's behavioral mode. Example Set: `player.cameraMode(Enum.CameraMode.Follow)` Example Get: `varol currentMode = player.cameraMode()`
    *   `player.controlEnabled(Boolean?)`: Disables (false) or enables (true) the player's movement controls. If called without arguments, returns whether controls are currently active.

**Examples:**
```lau
-- Location check
varol x, z = player.getCurrentTile()

if x ~= null then
	print("Player is on Tile -> X: " + x + " Z: " + z)
else
    player.alert("You are not standing on a tile!")
end

-- Camera manipulation
-- Check the current camera target
varol target = player.camera()

-- If the camera is not on the drone, move it to the drone!
if target ~= Enum.Camera.Drone then
    player.camera(Enum.Camera.Drone)
    player.cameraMode(Enum.CameraMode.Follow)
    
    -- Disable player movement while watching the drone
    player.controlEnabled(false)
end
```

### The `playerV2` Object
The Player V2 module introduces advanced events for market interactions and a daily gift system. Accessed with 'playerV2.' prefix.

*   **Daily Gifts**
    *   `playerV2.getGift()`: Attempts to claim the daily login gift. Displays a notification and opens the gift menu if successful.
*   **Interaction Events**
    *   These events allow your script to react to specific player actions in the game world and market.
    *   `playerV2.clicked:connect(func(button, x, z) ... end)`: Triggered when the player clicks in the world. Parameters: `Enum.ClickType`, `PositionX`, `PositionZ`. IMPORTANT: Only triggers on owned tiles or plants. Returns `null` for unowned areas.
    *   `playerV2.boughtSeed:connect(func(seed) ... end)`: Triggered when a seed is purchased. Returns the purchased `Enum.Seed` as a parameter.
    *   `playerV2.boughtGear:connect(func(gear) ... end)`: Triggered when a gear is purchased. Returns the purchased `Enum.Gear` as a parameter.
*   **UI and Navigation**
    *   `playerV2.mainScreenEnable(Boolean)`: Enables (true) or disables (false) the main screen UI.
    *   `playerV2.tpToDrone()`: Instantly teleports your character directly to the drone's current position.
    *   `playerV2.distanceToDrone()`: Returns the numerical distance between your character and the drone.
*   **Leaderboard Data**
    *   `playerV2.getScrapLeaderboardRank()`: Returns your current rank on the Top 50 Scrap Leaderboard as a number. If you are not in the top 50, it returns `null`.

**Examples:**
```lau
-- React to world clicks
playerV2.clicked:connect(func(button, x, z)
    if x ~= null then
        print("Clicked on: " + x + ", " + z)
    end
end)

-- Track seed purchases
playerV2.boughtSeed:connect(func(seed)
    if seed == Enum.Seed.Apple then
        print("Bought Apple seed")
    end
end)

-- Track gear purchases
playerV2.boughtGear:connect(func(gear)
    print("New gear added to inventory: " + gear)
end)

-- Check distance to drone, teleport if it is too far away
varol dist = playerV2.distanceToDrone()
if dist > 50 then
    playerV2.tpToDrone()
    print("Teleported to drone! Distance was: " + dist)
end

-- Check Scrap Leaderboard Rank
varol rank = playerV2.getScrapLeaderboardRank()
if rank ~= null then
    print("I am currently rank " + rank + " on the leaderboard!")
else
    print("I need more scrap to reach the top 50!")
end
```

### The `market` Object
Handles purchasing seeds, selling items, and market events.

*   **Market Data**
    *   `market.getSeedStock()`: Returns current seed stock.
    *   `market.getSeedPrice(Enum.Seed.[Type])`: Returns the price of a specific seed.
    *   `market.getSeedStockTime()` / `market.getGearStockTime()`: Returns time remaining for seed or gear stock refresh.
    *   `market.whatValue(slotNumber)`: Returns the market value of the item in the specified inventory slot.
*   **Transactions**
    *   `market.buySeed(Enum.Seed.[Type])` / `market.buyGear(Enum.Gear.[GearType])`: Buys a specific seed or gear. Some runtimes return `false` when the item cannot be bought.
    *   `market.sellItem(slotNumber)`: Sells the item in the specified inventory slot.
    *   `market.sellAllItem()`: Sells all sellable items from the inventory at once.
*   **Events**
    *   `market.changedSeedStock:connect(func() ... end)`: Triggered immediately when seed stocks refresh.
    *   `market.changedGearStock:connect(func() ... end)`: Triggered immediately when gear stocks refresh.

Example bounded purchase loop:
```lau
varol maxBuys = 20
while maxBuys > 0 do
    if market.buyGear(Enum.Gear.Watering_Can) == false then break end
    maxBuys -= 1
    task.wait(0.05)
end
```

### The `garden` Object
Provides functions for scanning the farm and retrieving plant data.

*   **Scanning the Garden**
    *   `garden.getGardenPositions()`: Scans the entire garden and returns a dictionary list of all active plants. The keys are the string coordinates (e.g., `"X,Z"`) and the values are strictly the `PlantName`s.
    *   `garden.getPlantEnum(Enum.Seed)`: Scans the field and lists ONLY the plants that match the specified seed type (e.g., `Enum.Seed.Apple`). Returns detailed data for each matched plant.
*   **Coordinate Specific Data**
    *   `garden.getPlantPosition(x, z)`: Returns highly detailed data about the plant at the specified X and Z coordinates. The returned object contains properties like `PlantName`, `PlantWeight`, `PlantPercent`, and if it bears fruit, it also includes `HasFruit`, `FruitName`, `FruitWeight`, `FruitPercent`.

When using `garden.getGardenPositions()`, validate keys before using them as table indexes. Table indexes must be numbers or strings.

The current code uses `garden.getGardenPositions()` as an active sweep driver. Because it returns coordinate-string keys, build a key-to-coordinate map during config initialization:
```lau
varol COORD_BY_KEY = {}
varol ALL_COORDS = {}

varol k = stateKey(x, z)
varol coord = {x, z, k}
COORD_BY_KEY[k] = coord
ALL_COORDS[#ALL_COORDS + 1] = coord
```

Then active sweeps can iterate only occupied garden positions and avoid parsing strings:
```lau
varol gp = garden.getGardenPositions()
for k, n inpairs(gp) do
    varol cd = COORD_BY_KEY[k]
    if cd then
        moveTo(cd[1], cd[2])
    end
end
```

Do not assume `garden.getGardenPositions()` includes growth percent. In current testing it returns keys and plant names only. Use `garden.getPlantPosition(x, z)` or drone APIs when growth percent is needed.

### The `task` Object
Provides utility functions for time and yielding. Note that loops do *not* strictly require yielding to prevent crashes, but `task.wait()` is available if needed.

*   `task.wait(seconds)`: Pauses the script for the specified number of seconds.
*   `task.date()`: Returns the current date and time as a string.
*   `task.clock()`: Returns a high-precision timestamp (useful for benchmarking code performance: `varol start = task.clock()`).

## Enums

### `Enum.Seed`
Available seed types for planting and purchasing:
`Apple`, `Bamboo`, `Banana`, `Blueberry`, `Bush`, `Cacao`, `Cactus`, `Carrot`, `Coconut`, `Corn`, `Dragon`, `Garlic`, `Glttch`, `Grape`, `Kiwi`, `Lemon`, `Lotus`, `Mango`, `Mushroom`, `Onion`, `Pear`, `Pepper`, `Pineapple`, `Pomegranate`, `Potato`, `Pumpkin`, `Strawberry`, `Tomato`, `Tree`, `Watermelon`, `Wheat`

*(Note: Use the standard seed name like `Enum.Seed.Apple`. Variations like `Enum.Seed.AppleTree` are incorrect and will not work).*

### `Enum.Direction`
Used for drone movement. There are exactly 4 available directions:
`Enum.Direction.North`, `Enum.Direction.East`, `Enum.Direction.South`, `Enum.Direction.West`.

### `Enum.DroneStatus`
Used to check if the drone is ready for a command, especially in async mode.
*   `Enum.DroneStatus.Busy`: The drone is currently performing an action (like moving or flipping).
*   `Enum.DroneStatus.Sleep`: The drone is idle and ready to receive a new command.

### `Enum.Gear`
Represents tools or machines.
Known gear enum names used by this project:
*   `Enum.Gear.Fertilizer`
*   `Enum.Gear.Lightning_Rod`
*   `Enum.Gear.Sprinkler`
*   `Enum.Gear.SprinklerV2`
*   `Enum.Gear.SprinklerV3`
*   `Enum.Gear.Watering_Can`: Used to manually water tiles via `drone.useItem()`.

## Built-in Functions
*   `print("Message")`: Prints text to the console.
*   `tonumber(string)`: Converts a string to a numeric value.
*   `req("Module.laum")`: Loads a module file and returns its exported value.

### The `string` Module
*   `string.find(str, substring)`: Returns the starting index of the substring within the string (1-indexed).
*   `string.match(str, pattern)`: Returns a match for the pattern, if supported by the runtime.
*   `string.sub(str, startIndex)`: Returns a substring starting from the specified index.

### The `math` Module
*   `math.random(min, max)`: Generates a random integer between the two specified numbers.
*   `math.round(number)`: Rounds a decimal number to the nearest integer (e.g., 4.6 -> 5).
*   `math.abs(number)`: Returns the absolute value (positive form) of the number. Useful for distances.
*   `math.pi`: Returns the mathematical constant Pi (3.1415...).

### The `list` Module
*   `list.find(listObject, item)`: Searches for `item` in `listObject` and returns its index (1-based). If the item is not found, it returns `null`.

## Automation Patterns From This Project

### Safe Drone Movement Wrapper
Direct movement with `droneV2.goto()` is fast, but in `--!ndrone` mode you should wait before and after physical commands:
```lau
func waitDrone()
    while drone.status() ~= Enum.DroneStatus.Sleep do
        task.wait(0.05)
    end
end

func moveTo(x, z)
    waitDrone()
    droneV2.goto(x, z)
    waitDrone()
end
```

### Field State Keys
Use stable coordinate strings for field-state tables:
```lau
func stateKey(x, z)
    return x + "," + z
end

varol key = stateKey(-1, 3)
fieldState[key] = "empty"
```

Avoid indexing field-state tables with plant objects, coordinate objects, or other tables.

### Seed Inventory Matching
Inventory items can use different names for seeds. This project checks several fields:
```lau
if item AND (item.Type == "Seed" OR item.Type == "Seeds") then
    if item.Name == "Lotus" OR item.OriginalKey == "Lotus" then
        print("Found Lotus seeds")
    end
end
```

### Market Refresh Automation
Use market events for immediate buying instead of checking stock timers only after a long sweep:
```lau
varol buyingGears = false

market.changedGearStock:connect(func()
    if buyingGears then return end
    buyingGears = true
    varol maxBuys = 20
    while maxBuys > 0 do
        if market.buyGear(Enum.Gear.Watering_Can) == false then break end
        maxBuys -= 1
        task.wait(0.05)
    end
    buyingGears = false
end)
```

## Common Techniques and Tips From Community Scripts

Community scripts often use older APIs or deliberately unsafe shortcuts for speed. Treat these as patterns to learn from, not rules to copy blindly. In this game, some drone API checks are expensive enough that the fastest script is sometimes the one with fewer safety checks.

### Dynamic Plot Size From Player Land
Instead of hardcoding the farm size, derive it from the player's land upgrade:
```lau
varol MAP_SIZE = (player.getTileNumber() * 2) - 1
varol FROM_CENTRE = player.getTileNumber() - 1
```

This works well for simple square farms centered around `(0, 0)`. For scripts using a configured grid, keep the explicit config value if it is easier to reason about.

### Manual Step Movement Without `droneV2.goto`
Older scripts often move by comparing current position and stepping one tile at a time:
```lau
varol D = Enum.Direction

func moveto(x, z)
    varol done = false
    while done == false do
        varol cx = drone.getPositionX()
        varol cz = drone.getPositionZ()
        if cx == x AND cz == z then done = true end
        if cx > x then
            drone.move(D.West)
        elseif cx < x then
            drone.move(D.East)
        end
        if cz > z then
            drone.move(D.North)
        elseif cz < z then
            drone.move(D.South)
        end
    end
    return done
end
```

This is portable and does not require `droneV2.goto`, but repeated position reads are expensive. Prefer `droneV2.goto(x, z)` when available and when direct movement is safe.

### Minimal Harvest/Crop Action
A compact action function can rely on capability checks instead of detailed plant inspection:
```lau
func harvestAny()
    if drone.canCrop() then
        return drone.crop()
    elseif drone.canHarvest() then
        return drone.harvest()
    end
end
```

This avoids `drone.getPlant()` when the exact plant type is not important. Use this only when cropping the current plant is acceptable. If fruit trees must be preserved, keep a fruit/crop distinction.

### Inventory Normalization
Inventory seed items may expose their enum through different fields. Community scripts often convert seed names or original keys back to `Enum.Seed`:
```lau
varol item = player.getInventory()[i]
if item["Type"] == "Seed" then
    varol e = Enum.Seed[item["OriginalKey"]]
    if e == null then
        e = Enum.Seed[item["Name"]]
    end
end
```

Some scripts use `pcall` and `tostring` to handle odd item fields:
```lau
varol ok = item["OriginalKey"]
varol success = pcall(func()
    ok = tostring(ok)
end)
if success then
    item["Enum"] = Enum.Seed[ok]
else
    item["Enum"] = Enum.Seed[item["Name"]]
end
```

Use this pattern when item names are inconsistent. In stable scripts, explicit seed maps are easier to audit.

### Budgeted Seed Purchasing
Seed buying can be based on available scrap and market stock:
```lau
varol budget = player.scrap() * 0.8
varol stock = market.getSeedStock()

for i, row inpairs(stock) do
    varol seed = row["Seed"]
    varol price = market.getSeedPrice(seed)
    if price <= budget then
        market.buySeed(seed)
    end
end
```

The safer version also caps buy count and yields during long purchase loops:
```lau
varol bought = 0
while bought < 20 do
    if market.buySeed(seed) == false then break end
    bought += 1
    task.wait(0.05)
end
```

### Restock Timer Polling
Before market restock events were commonly used, scripts polled restock timers:
```lau
varol currentStock = market.getSeedStock()
varol lastRestockTime = market.getSeedStockTime()
varol lastCheck = task.clock()

if (task.clock() - lastCheck) >= lastRestockTime then
    currentStock = market.getSeedStock()
    lastRestockTime = market.getSeedStockTime()
    lastCheck = task.clock()
end
```

Prefer `market.changedSeedStock:connect` when available. Timer polling is still useful as a fallback or for scripts that start before event support is unlocked.

### Function Tables as Local Modules
Single-file scripts often store actions in a dictionary:
```lau
varol functions = {}

functions["h"] = func()
    if drone.canHarvest() then
        drone.harvest()
    end
end

functions["h"]()
```

This is a lightweight alternative to `.laum` modules. For larger scripts, separate modules are easier to maintain and paste into the runtime.

### Recursive or Self-Restarting Loops
Some community scripts restart a full pass by calling the same function again:
```lau
functions["start"] = func()
    -- full field pass
    return functions["start"]()
end
```

This is compact, but an explicit `while true do` loop is usually clearer and less risky. Use recursion only if you know the Lau runtime does not grow the call stack for that pattern.

### Simple Snake Movement
A standard snake path alternates horizontal direction by row:
```lau
for row = 1, GRID do
    for col = 1, GRID do
        if col < GRID then
            if row % 2 == 1 then
                drone.move(Enum.Direction.East)
            else
                drone.move(Enum.Direction.West)
            end
        end
    end
    if row < GRID then
        drone.move(Enum.Direction.South)
    end
end
```

This covers every tile and is easy to debug. For fully planted farms, a simpler continuous lane loop can be faster because it removes row/column state checks.

### Continuous Lane Loop
For farms laid out as long lines, a minimal loop can move repeatedly in one direction, then shift lanes:
```lau
varol plotsize = 27
while true do
    for j = 1, plotsize do
        drone.move(Enum.Direction.South)
        if drone.canHarvest() then
            drone.harvest()
        end
    end
    drone.move(Enum.Direction.East)
end
```

This is very fast when the field layout matches the path and the drone can wrap or return naturally. It is fragile if the farm has mixed crops, empty spots, or plants that should not be cropped.

### Negative-Step `for` Loops
Lau supports numeric `for` loops with a negative step:
```lau
for i = PlotSize, -PlotSize, -1 do
    drone.move(Enum.Direction.North)
end
```

This can simplify backtracking paths and remove separate reverse-loop logic.

### Swap Pathing With `droneV2.swap`
Community swap modules move a plant by repeatedly swapping and stepping along an axis:
```lau
for i = xFirst, xSecond - xStep, xStep do
    droneV2.swap(xMoveDir)
    task.wait(0.1)
    drone.move(xMoveDir)
    task.wait(0.1)
end
```

The general technique is:
1. Move the drone to the first plant.
2. Determine X and Z directions plus reverse directions.
3. Swap and move along one axis.
4. Swap and move along the other axis.
5. Walk back with reverse swaps to place the displaced plant.

Use small waits between `swap` and `move`; those are physical actions and can be ignored if issued too quickly.

### Mid-Loop Selling
Some scripts sell when inventory appears full:
```lau
if player.getInventorySize() >= player.getFruitCapacity() then
    market.sellAllItem()
end
```

This prevents lost fruit in capacity-limited runs, but checking inventory every tile adds overhead. For high-speed routes, prefer timed autosell, event-driven selling if available, or less frequent checks.

### `player.sent` vs `player.alert`
Community scripts sometimes use `player.sent("message")` for chat-style bot output:
```lau
player.sent("[Bot] Loop done")
```

Use `player.alert()` for short UI alerts and `player.sent()` only if the runtime supports it and chat output is desired. Frequent player messages slow scripts and clutter output.

### Daily Gift Claiming
If the Player V2 unlock is available, scripts can try to claim a daily gift at startup:
```lau
if playerV2.getGift() then
    player.sent("[Bot] Daily reward claimed!")
end
```

Keep this optional; it is unrelated to farming throughput and may require a separate unlock.

### Performance Tips From Community Testing
Common observations from shared scripts:
*   `drone.getPositionX()` / `drone.getPositionZ()` checks add overhead when done every tile.
*   `drone.canHarvest()` and `drone.canCrop()` are usually cheaper than detailed plant inspection, but still cost time.
*   `drone.getPlant()` plus list lookups are useful for correctness, but can be slower than simply trying the desired action.
*   Selling every tile or checking inventory every tile is usually too expensive for high-throughput routes.
*   If a farm is fully planted with the same plant type, repeated action spam can outperform careful checks.
*   If a farm is mixed or valuable fruit trees must be preserved, use safer checks even if they are slower.

The practical rule is: choose checks based on the cost of being wrong. On disposable crop fields, `crop()` then `plant()` can be acceptable. On fruit-tree farms, accidental `crop()` can destroy value, so keep fruit-safe logic.

## The `display` Module

The Display module allows you to create custom User Interfaces (UI) on the screen using code.

### 1. Main Screen Initialization
`display.mainScreen()`: Returns the unique ID of the player's main screen UI.

You MUST set the `Parent` of any new UI elements you create to this mainScreen ID. If you don't, they will not be visible on the screen!

### 2. Creating UI Elements
`display.newUI(Enum.UIElement)`: Creates a new UI element and returns its unique ID.

This function requires an `Enum.UIElement` value to know what to build.

**Available Enums:** `.Box`, `.Label`, `.ImageLabel`, `.Button`, `.ImageButton`, `.Input`

### 3. UI Limits & Security (Crucial)
To prevent screen lag and malicious spam, strict limits are enforced per script. You can only create up to:

*   **Box:** Max 20 elements
*   **Label & ImageLabel:** Max 10 elements each
*   **Button & ImageButton:** Max 5 elements each
*   **Input:** Max 3 elements

**Security Warning:** The engine tracks UI creation strictly. You cannot rapidly create and delete UI elements to bypass limits. Spamming UI creations is blocked and will throw errors.

#### Example code creating a basic UI
```lau
-- 1. Get the main screen
varol screen = display.mainScreen()

-- 2. Create a new Box element
varol myBox = display.newUI(Enum.UIElement.Box)

myBox.Parent = screen
myBox.Position = udim2(0.5, 0, 0.5, 0)
myBox.Size = udim2(0.2, 0.2)

-- 3. Create a new Label element
varol myLabel = display.newUI(Enum.UIElement.Label)

myLabel.Parent = myBox
myLabel.Size = udim2(1, 0.1)

myLabel.Text = "Hello UI"

print("UI created successfully!")
```

## The `http` Module

The HTTP module allows your scripts to communicate with external web services. Access is restricted to whitelisted domains for security. Accessed with `http.` prefix.

### 1. Network Requests
*   `http.get(url)`: Sends a GET request to a whitelisted URL and returns the response as a string.
*   `http.post(url, data)`: Sends a POST request with string data to a whitelisted URL.
*   `http.request(url/string, method/string, body/string, headers/list)`: Sends a fully customizable network request. Essential for communicating with AI APIs or services that require authentication (API Keys) via headers.

**Security Note:** Methods are strictly limited to `GET` and `POST`. If sending a GET request, pass an empty string (`""`) for the body parameter.

### 2. JSON Processing
Used to convert between Lau Lists and JSON strings. These functions have internal memory limits to prevent system overloads.

*   `http.jsonDecode(jsonString)`: Converts a JSON string into a Lau List.
*   `http.jsonEncode(list)`: Converts a Lau List into a JSON string.

### 3. System Limits & Security (Crucial)
To ensure server stability, the following strict limits are enforced on all HTTP operations:

*   **Rate Limit (Standard):** 70 requests per minute.
*   **Rate Limit (Private Server):** 200 requests per minute.
*   **Timeout:** The engine waits a maximum of 5 seconds for a response before cancelling.
*   **Whitelist:** Requests only work with pre-approved, whitelisted domains. Unrecognized URLs will throw an error.

#### Safe HTTP request example
```lau
-- Using pcall is highly recommended for HTTP due to timeouts/limits
varol success, response = pcall(http.get, "https://api.whitelisted.com/data")

if success == true then
    varol data = http.jsonDecode(response)
    print("Received: " + data.name)
else
    player.alert("HTTP Request Failed: Time-out or Limit reached.")
end
```

#### Advanced API requests example
```lau
-- Using pcall is highly recommended for HTTP due to timeouts/limits
varol success, response = pcall(http.get, "https://api.whitelisted.com/data")

if success == true then
    varol data = http.jsonDecode(response)
    print("Received: " + data.name)
else
    player.alert("HTTP Request Failed: Time-out or Limit reached.")
end
```

### Whitelisted Domains
*   `generativelanguage.googleapis.com` (Gemini)
*   `api.openai.com` (OpenAI)
*   `api-inference.huggingface.co`
*   `api.groq.com` (High-speed LLM)
*   `api.anthropic.com` (Claude)
*   `api.deepseek.com`
*   `api.aleph-alpha.com`
*   `openrouter.ai`
*   `pollinations.ai`
*   `script.google.com` (Google Apps Script)
*   `script.googleusercontent.com` (Apps Script Content)
*   `www.googleapis.com` (YouTube Data & Google Services)
*   `github.com` & `raw.githubusercontent.com`
*   `api.gitlab.com`
*   `pastebin.com`
*   `replit.dev`
*   `api.vercel.com` & `api.render.com`
*   `api.cloudflare.com`
*   `backboard.railway.app`
*   `api.minecraftservices.com`
*   `api.imgur.com` & `api.giphy.com`
*   `webhook.lewisakura.moe` (Discord Proxy)
*   `api.telegram.org`
*   `hooks.slack.com`
*   `api.sendgrid.com` (Email)
*   `pusher.com`
*   `api.open-meteo.com`
*   `worldtimeapi.org`
*   `newsapi.org`
*   `api.nasa.gov`
*   `us1.locationiq.com`
*   `themealdb.com` & `catfact.ninja`
*   `en.wikipedia.org`
