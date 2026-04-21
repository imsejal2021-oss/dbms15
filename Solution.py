
from Planner import TablePlan, SelectPlan, ProjectPlan, ProductPlan
from RelationalOp import Predicate, Term, Expression, Constant, SelectScan, ProjectScan, ProductScan
from Record import Schema, Layout, TableScan, RecordID
from Metadata import MetadataMgr
from Transaction import Transaction


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_term_fields(term):
    """
    Return (lhs_field, rhs_field, lhs_const, rhs_const) for a Term.
    Fields are strings; constants are Constant objects (or None).
    """
    lhs_field = None
    rhs_field = None
    lhs_const = None
    rhs_const = None

    lhs_val = term.lhs.exp_value
    rhs_val = term.rhs.exp_value

    if isinstance(lhs_val, Constant):
        lhs_const = lhs_val
    else:
        lhs_field = lhs_val  # field name string

    if isinstance(rhs_val, Constant):
        rhs_const = rhs_val
    else:
        rhs_field = rhs_val  # field name string

    return lhs_field, rhs_field, lhs_const, rhs_const


def _schema_has_field(schema, field_name):
    return field_name in schema.field_info


def _term_applies_to(term, schema):
    """True if ALL fields referenced by this term exist in schema."""
    lhs_field, rhs_field, lhs_const, rhs_const = _get_term_fields(term)
    if lhs_field and not _schema_has_field(schema, lhs_field):
        return False
    if rhs_field and not _schema_has_field(schema, rhs_field):
        return False
    return True


def _make_predicate(terms):
    pred = Predicate()
    for t in terms:
        pred.terms.append(t)
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# 1. BetterQueryPlanner — selection pushdown + join reordering
# ─────────────────────────────────────────────────────────────────────────────

class BetterQueryPlanner:
    """
    Optimized query planner with selection pushdown and join reordering.
    """

    def __init__(self, mm):
        self.mm = mm

    def createPlan(self, tx, query_data):
        tables = query_data['tables']
        all_terms = list(query_data['predicate'].terms)

        # Step 1: Build TablePlan for each table
        table_plans = {}
        for table_name in tables:
            table_plans[table_name] = TablePlan(tx, table_name, self.mm)

        # Step 2: Push selections — wrap each TablePlan with terms that
        #         reference only that table's fields
        pushed_plans = {}
        remaining_terms = []

        for table_name, tp in table_plans.items():
            schema = tp.plan_schema()
            local_terms = []
            for term in all_terms:
                if _term_applies_to(term, schema):
                    local_terms.append(term)

            if local_terms:
                pred = _make_predicate(local_terms)
                pushed_plans[table_name] = SelectPlan(tp, pred)
            else:
                pushed_plans[table_name] = tp

        # Identify join terms: terms that reference fields from 2 tables
        # We'll apply these after the join
        for term in all_terms:
            lhs_field, rhs_field, lhs_const, rhs_const = _get_term_fields(term)
            is_join = lhs_field is not None and rhs_field is not None
            if not is_join and lhs_const is None and rhs_const is None:
                remaining_terms.append(term)

        # Step 3: Join ordering — start with smallest table, greedily pick
        #         next table that shares a join condition with current result
        table_names = list(tables)

        # Sort by estimated output size (ascending)
        table_names.sort(key=lambda t: pushed_plans[t].recordsOutput() or 1)

        current_plan = pushed_plans[table_names[0]]
        current_schema = current_plan.plan_schema()
        joined = {table_names[0]}
        remaining_tables = list(table_names[1:])

        while remaining_tables:
            # Find table with a join term connecting to current schema
            best_table = None
            best_terms = []

            for candidate in remaining_tables:
                cand_schema = pushed_plans[candidate].plan_schema()
                join_terms = []
                for term in all_terms:
                    lhs_field, rhs_field, lhs_const, rhs_const = _get_term_fields(term)
                    if lhs_field and rhs_field:
                        # Both fields must exist: one in current, one in candidate
                        in_current = _schema_has_field(current_schema, lhs_field) or \
                                     _schema_has_field(current_schema, rhs_field)
                        in_cand    = _schema_has_field(cand_schema, lhs_field) or \
                                     _schema_has_field(cand_schema, rhs_field)
                        if in_current and in_cand:
                            join_terms.append(term)

                if join_terms:
                    # Prefer connected table with smallest size
                    if best_table is None or \
                       (pushed_plans[candidate].recordsOutput() or 1) < \
                       (pushed_plans[best_table].recordsOutput() or 1):
                        best_table = candidate
                        best_terms = join_terms

            if best_table is None:
                # No connected table found — just pick the smallest remaining
                best_table = min(remaining_tables,
                                 key=lambda t: pushed_plans[t].recordsOutput() or 1)
                best_terms = []

            remaining_tables.remove(best_table)
            joined.add(best_table)

            # Build product then immediately apply join conditions
            current_plan = ProductPlan(current_plan, pushed_plans[best_table])

            if best_terms:
                join_pred = _make_predicate(best_terms)
                current_plan = SelectPlan(current_plan, join_pred)

            # Rebuild merged schema for next iteration
            current_schema = _merged_schema(current_plan)

        # Step 4: Apply any leftover terms (cross-table filter not yet applied)
        #         Build combined schema of all joined tables to check coverage
        all_joined_terms = []
        for term in all_terms:
            if _term_applies_to(term, current_schema):
                all_joined_terms.append(term)

        if all_joined_terms:
            final_pred = _make_predicate(all_joined_terms)
            current_plan = SelectPlan(current_plan, final_pred)

        # Step 5: Project to requested fields
        return ProjectPlan(current_plan, *query_data['fields'])


def _merged_schema(plan):
    """Walk down a plan tree to collect the merged schema."""
    return plan.plan_schema()


# ─────────────────────────────────────────────────────────────────────────────
# 2. BTreeIndex — in-memory B-tree for insert and search
# ─────────────────────────────────────────────────────────────────────────────

class _BTreeNode:
    """Internal node of the B-tree (in memory)."""

    def __init__(self, order):
        self.order = order          # max children = order
        self.keys = []              # list of key values
        self.values = []            # list of list[RecordID] (leaf) or None (internal)
        self.children = []          # list of _BTreeNode (internal only)
        self.is_leaf = True

    def is_full(self):
        return len(self.keys) >= self.order - 1


class BTreeIndex:
    """
    Simple B-tree index stored entirely in memory.
    Supports insert(key_value, record_id) and search(key_value) -> [RecordID].
    """

    ORDER = 64  # branching factor

    def __init__(self, tx, index_name, key_type, key_length):
        self.tx = tx
        self.index_name = index_name
        self.key_type = key_type
        self.key_length = key_length
        self.root = _BTreeNode(self.ORDER)

    # ── public API ────────────────────────────────────────────────────────────

    def insert(self, key_value, record_id):
        result = self._insert_recursive(self.root, key_value, record_id)
        if result is not None:
            # Root was split — create new root
            mid_key, new_node = result
            new_root = _BTreeNode(self.ORDER)
            new_root.is_leaf = False
            new_root.keys = [mid_key]
            new_root.children = [self.root, new_node]
            self.root = new_root

    def search(self, key_value):
        """Return list of RecordID objects matching key_value."""
        return self._search_recursive(self.root, key_value)

    def close(self):
        pass  # In-memory; nothing to close

    # ── internal helpers ──────────────────────────────────────────────────────

    def _insert_recursive(self, node, key, rid):
        if node.is_leaf:
            # Find insert position
            pos = self._find_pos(node.keys, key)
            # Check for existing key
            if pos < len(node.keys) and node.keys[pos] == key:
                node.values[pos].append(rid)
            else:
                node.keys.insert(pos, key)
                node.values.insert(pos, [rid])

            if node.is_full():
                return self._split_leaf(node)
            return None
        else:
            # Internal node — find correct child
            pos = self._find_pos(node.keys, key)
            result = self._insert_recursive(node.children[pos], key, rid)
            if result is not None:
                mid_key, new_child = result
                node.keys.insert(pos, mid_key)
                node.children.insert(pos + 1, new_child)
                if node.is_full():
                    return self._split_internal(node)
            return None

    def _split_leaf(self, node):
        mid = len(node.keys) // 2
        new_node = _BTreeNode(self.ORDER)
        new_node.is_leaf = True
        new_node.keys = node.keys[mid:]
        new_node.values = node.values[mid:]
        node.keys = node.keys[:mid]
        node.values = node.values[:mid]
        return new_node.keys[0], new_node

    def _split_internal(self, node):
        mid = len(node.keys) // 2
        mid_key = node.keys[mid]
        new_node = _BTreeNode(self.ORDER)
        new_node.is_leaf = False
        new_node.keys = node.keys[mid + 1:]
        new_node.children = node.children[mid + 1:]
        node.keys = node.keys[:mid]
        node.children = node.children[:mid + 1]
        return mid_key, new_node

    def _search_recursive(self, node, key):
        pos = self._find_pos(node.keys, key)
        if node.is_leaf:
            if pos < len(node.keys) and node.keys[pos] == key:
                return list(node.values[pos])
            return []
        else:
            return self._search_recursive(node.children[pos], key)

    @staticmethod
    def _find_pos(keys, key):
        lo, hi = 0, len(keys)
        while lo < hi:
            mid = (lo + hi) // 2
            if keys[mid] < key:
                lo = mid + 1
            else:
                hi = mid
        return lo


# ─────────────────────────────────────────────────────────────────────────────
# 3. CompositeIndex — multi-field index backed by a BTree on tuple keys
# ─────────────────────────────────────────────────────────────────────────────

class CompositeIndex:
    """
    Index over multiple fields. Keys are tuples of values.
    Internally uses the same BTree structure.
    """

    def __init__(self, tx, index_name, field_names, field_types, field_lengths):
        self.tx = tx
        self.index_name = index_name
        self.field_names = field_names   # tuple of str
        self.field_types = field_types   # tuple of str ('int'/'str')
        self.field_lengths = field_lengths
        self._index = {}                 # dict: tuple(values) -> [RecordID]

    def insert(self, field_values, record_id):
        """field_values: tuple of values in same order as field_names."""
        key = tuple(field_values)
        if key not in self._index:
            self._index[key] = []
        self._index[key].append(record_id)

    def search(self, field_values):
        """Return list of RecordID matching exact tuple of values."""
        key = tuple(field_values)
        return list(self._index.get(key, []))

    def close(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 4. IndexScan — iterates records via an index
# ─────────────────────────────────────────────────────────────────────────────

class IndexScan:
    """
    A scan that retrieves RecordIDs from an index and positions the
    underlying TableScan at each matching record.
    """

    def __init__(self, table_scan, index, search_key):
        self.table_scan = table_scan
        self.index = index
        self.search_key = search_key
        self._rids = []
        self._pos = -1
        self._fetch_rids()

    def _fetch_rids(self):
        if isinstance(self.index, CompositeIndex):
            self._rids = self.index.search(self.search_key)
        else:
            self._rids = self.index.search(self.search_key)

    def nextRecord(self):
        self._pos += 1
        if self._pos >= len(self._rids):
            return False
        rid = self._rids[self._pos]
        self.table_scan.moveToRecordID(rid)
        return True

    def getInt(self, field_name):
        return self.table_scan.getInt(field_name)

    def getString(self, field_name):
        return self.table_scan.getString(field_name)

    def getVal(self, field_name):
        return self.table_scan.getVal(field_name)

    def hasField(self, field_name):
        return self.table_scan.hasField(field_name)

    def beforeFirst(self):
        self._pos = -1

    def closeRecordPage(self):
        self.table_scan.closeRecordPage()


# ─────────────────────────────────────────────────────────────────────────────
# 5. IndexPlan — a Plan node that wraps an IndexScan
# ─────────────────────────────────────────────────────────────────────────────

class IndexPlan:
    """
    A Plan that opens an IndexScan for a given table + index + search key.
    Used by IndexQueryPlanner in place of a plain TablePlan.
    """

    def __init__(self, tx, table_name, mm, index, search_key):
        self.tx = tx
        self.table_name = table_name
        self.mm = mm
        self.index = index
        self.search_key = search_key
        self.layout = mm.getLayout(tx, table_name)
        self._stat = mm.getStatInfo(tx, table_name, self.layout)

    def open(self):
        ts = TableScan(self.tx, self.table_name, self.layout)
        return IndexScan(ts, self.index, self.search_key)

    def blocksAccessed(self):
        return 1  # index lookup is O(1) block accesses effectively

    def recordsOutput(self):
        return 1  # equality lookup typically returns very few rows

    def distinctValues(self, field_name):
        return self._stat.get('distinctValues', 1)

    def plan_schema(self):
        return self.layout.schema


# ─────────────────────────────────────────────────────────────────────────────
# 6. IndexQueryPlanner — routes equality predicates through indexes
# ─────────────────────────────────────────────────────────────────────────────

class IndexQueryPlanner:
    """
    Planner that checks equality predicates against available indexes.
    If an index match exists for a table, it substitutes an IndexPlan.
    Falls back to BetterQueryPlanner (or BasicQueryPlanner) for unindexed tables.
    """

    def __init__(self, mm, indexes, better_planner=None):
        self.mm = mm
        self.indexes = indexes or {}    # {table_name: {field_key: IndexObject}}
        self.better_planner = better_planner

    def createPlan(self, tx, query_data):
        tables = query_data['tables']
        all_terms = list(query_data['predicate'].terms)

        # Build per-table plans, using indexes where possible
        table_plans = {}
        used_terms = set()

        for table_name in tables:
            table_idx = self.indexes.get(table_name, {})
            layout = self.mm.getLayout(tx, table_name)
            schema = layout.schema

            index_plan = None

            # Try single-field BTreeIndex first
            for field_key, idx_obj in table_idx.items():
                if isinstance(field_key, tuple):
                    continue  # handle composite separately below
                for i, term in enumerate(all_terms):
                    lhs_field, rhs_field, lhs_const, rhs_const = _get_term_fields(term)
                    # equality: field = constant
                    if lhs_field == field_key and rhs_const is not None:
                        search_key = rhs_const.const_value
                        index_plan = IndexPlan(tx, table_name, self.mm, idx_obj, search_key)
                        used_terms.add(i)
                        break
                    elif rhs_field == field_key and lhs_const is not None:
                        search_key = lhs_const.const_value
                        index_plan = IndexPlan(tx, table_name, self.mm, idx_obj, search_key)
                        used_terms.add(i)
                        break
                if index_plan:
                    break

            # Try composite index
            if index_plan is None:
                for field_key, idx_obj in table_idx.items():
                    if not isinstance(field_key, tuple):
                        continue
                    field_names = field_key
                    search_vals = []
                    matched_terms = []
                    for fname in field_names:
                        matched = None
                        for i, term in enumerate(all_terms):
                            lhs_field, rhs_field, lhs_const, rhs_const = _get_term_fields(term)
                            if lhs_field == fname and rhs_const is not None:
                                matched = (rhs_const.const_value, i)
                                break
                            elif rhs_field == fname and lhs_const is not None:
                                matched = (lhs_const.const_value, i)
                                break
                        if matched:
                            search_vals.append(matched[0])
                            matched_terms.append(matched[1])
                        else:
                            break
                    if len(search_vals) == len(field_names):
                        index_plan = IndexPlan(tx, table_name, self.mm, idx_obj, tuple(search_vals))
                        used_terms.update(matched_terms)
                        break

            if index_plan:
                table_plans[table_name] = index_plan
            else:
                table_plans[table_name] = TablePlan(tx, table_name, self.mm)

        # Now assemble joins — use better_planner's ordering if available,
        # otherwise simple left-to-right join
        if self.better_planner is not None:
            # Delegate join ordering to BetterQueryPlanner but substitute
            # our index plans for the table plans it would create
            return self._create_plan_with_better(tx, query_data, table_plans)
        else:
            return self._create_plan_simple(tx, query_data, table_plans, used_terms)

    def _create_plan_with_better(self, tx, query_data, table_plans):
        """Use BetterQueryPlanner's join logic but inject our index plans."""
        tables = query_data['tables']
        all_terms = list(query_data['predicate'].terms)

        # Sort tables smallest first (same as BetterQueryPlanner)
        table_names = list(tables)
        table_names.sort(key=lambda t: table_plans[t].recordsOutput() or 1)

        current_plan = table_plans[table_names[0]]
        current_schema = current_plan.plan_schema()
        remaining_tables = list(table_names[1:])

        while remaining_tables:
            best_table = None
            best_terms = []

            for candidate in remaining_tables:
                cand_schema = table_plans[candidate].plan_schema()
                join_terms = []
                for term in all_terms:
                    lhs_field, rhs_field, lhs_const, rhs_const = _get_term_fields(term)
                    if lhs_field and rhs_field:
                        in_current = _schema_has_field(current_schema, lhs_field) or \
                                     _schema_has_field(current_schema, rhs_field)
                        in_cand    = _schema_has_field(cand_schema, lhs_field) or \
                                     _schema_has_field(cand_schema, rhs_field)
                        if in_current and in_cand:
                            join_terms.append(term)
                if join_terms:
                    if best_table is None or \
                       (table_plans[candidate].recordsOutput() or 1) < \
                       (table_plans[best_table].recordsOutput() or 1):
                        best_table = candidate
                        best_terms = join_terms

            if best_table is None:
                best_table = min(remaining_tables,
                                 key=lambda t: table_plans[t].recordsOutput() or 1)
                best_terms = []

            remaining_tables.remove(best_table)
            current_plan = ProductPlan(current_plan, table_plans[best_table])

            if best_terms:
                join_pred = _make_predicate(best_terms)
                current_plan = SelectPlan(current_plan, join_pred)

            current_schema = current_plan.plan_schema()

        # Apply any remaining filter terms
        all_joined_terms = [t for t in all_terms
                            if _term_applies_to(t, current_schema)]
        if all_joined_terms:
            current_plan = SelectPlan(current_plan, _make_predicate(all_joined_terms))

        return ProjectPlan(current_plan, *query_data['fields'])

    def _create_plan_simple(self, tx, query_data, table_plans, used_terms):
        """Fallback: simple left-to-right join with full predicate at end."""
        tables = list(query_data['tables'])
        all_terms = list(query_data['predicate'].terms)

        plan = table_plans[tables[0]]
        for table_name in tables[1:]:
            plan = ProductPlan(plan, table_plans[table_name])

        remaining = [t for i, t in enumerate(all_terms) if i not in used_terms]
        if remaining:
            plan = SelectPlan(plan, _make_predicate(remaining))

        return ProjectPlan(plan, *query_data['fields'])


# ─────────────────────────────────────────────────────────────────────────────
# 7. create_indexes — build and populate all indexes
# ─────────────────────────────────────────────────────────────────────────────

def create_indexes(db, tx, index_defs=None, composite_index_defs=None):
    """
    Build BTreeIndex and CompositeIndex objects and populate them by
    scanning each table once.

    Returns:
        dict {table_name: {field_key: IndexObject}}
        - field_key is a str for BTreeIndex
        - field_key is a tuple of field names for CompositeIndex
    """
    index_defs = index_defs or {}
    composite_index_defs = composite_index_defs or {}

    # Collect all tables that need indexing
    all_tables = set(list(index_defs.keys()) + list(composite_index_defs.keys()))

    # Build empty index objects
    indexes = {}

    for table_name in all_tables:
        indexes[table_name] = {}

        # BTree indexes for this table
        for (field_name, field_type, field_length) in index_defs.get(table_name, []):
            idx_name = f"idx_{table_name}_{field_name}"
            idx = BTreeIndex(tx, idx_name, field_type, field_length)
            indexes[table_name][field_name] = idx

        # Composite indexes for this table
        for (field_names, field_types, field_lengths) in composite_index_defs.get(table_name, []):
            idx_name = "idx_" + table_name + "_" + "_".join(field_names)
            idx = CompositeIndex(tx, idx_name, field_names, field_types, field_lengths)
            indexes[table_name][tuple(field_names)] = idx

    # Populate indexes by scanning each table once
    for table_name in all_tables:
        layout = db.mm.getLayout(tx, table_name)
        ts = TableScan(tx, table_name, layout)

        btree_fields = {
            fname: indexes[table_name][fname]
            for fname in indexes[table_name]
            if isinstance(fname, str)
        }
        composite_fields = {
            fnames: indexes[table_name][fnames]
            for fnames in indexes[table_name]
            if isinstance(fnames, tuple)
        }

        while ts.nextRecord():
            rid = ts.currentRecordID()

            # Insert into each BTree index
            for field_name, idx in btree_fields.items():
                field_info = layout.schema.field_info.get(field_name, {})
                ftype = field_info.get('field_type', 'str')
                if ftype == 'int':
                    val = ts.getInt(field_name)
                else:
                    val = ts.getString(field_name)
                idx.insert(val, rid)

            # Insert into each composite index
            for field_names, idx in composite_fields.items():
                values = []
                for fname in field_names:
                    finfo = layout.schema.field_info.get(fname, {})
                    ftype = finfo.get('field_type', 'str')
                    if ftype == 'int':
                        values.append(ts.getInt(fname))
                    else:
                        values.append(ts.getString(fname))
                idx.insert(tuple(values), rid)

        ts.closeRecordPage()

    return indexes
